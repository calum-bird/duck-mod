"""Run a dance policy with no viewer, no display, and no GPU.

The first eval battery was built on upstream's `infer_policy.py` main loop,
which launches a GLFW **window**. On a training node that fails outright
("X11: The DISPLAY environment variable is missing"), and `MUJOCO_GL=egl` does
not help because the problem is the viewer, not the renderer. The deeper
mistake was renting an H100 to evaluate one duck: a single robot stepped at
50 Hz is a CPU workload, and evaluation should never have needed the GPU at all.

So this drives the policy directly. It still constructs upstream's
`PolicyInference` and calls its own `infer()` / `apply_action()`, so the 61-dim
observation is assembled by exactly the code that assembles it in the viewer and
on the robot -- reimplementing that here is the one shortcut that would quietly
invalidate every number the battery produces. Only the loop is ours:

    update the beat clock -> infer -> apply -> 4 physics steps -> record

which is the same sequence, minus the window.
"""

from __future__ import annotations

import os
from pathlib import Path

from microduck_dance.dance_driver import (
    BeatController,
    TrajectoryRecorder,
    load_infer_policy,
    make_dance_inference,
)

# Upstream's default scene (flat ground + the walking collision model).
MICRODUCK_SCENE = "src/mjlab_microduck/robot/microduck/scene.xml"

# The viewer runs the policy at 50 Hz over a 4-substep physics tick.
DECIMATION = 4

# XL330 torque constant (Nm/A) from the BAM m6 fit; used when the bam package
# is not importable (e.g. the CPU eval box). Kept in sync with
# bam.load_model("xl330", "m6").kt.value.
XL330_KT = 0.36601349688984386


def condition_model(model, current_limit: float = 1.75) -> None:
    """Apply the same model surgery upstream's viewer applies before inference.

    Two pieces, both discovered the expensive way — a freshly trained, healthy
    policy read as fallen-90%-of-the-run through a harness that skipped them:

    1. ``model.opt.timestep = 0.005``. The scene XML's native timestep is
       0.002, and the 50 Hz control contract is DECIMATION x 0.005. Stepping
       DECIMATION x native ran the policy at 125 Hz — every action applied
       2.5x too fast — which turned a clean bob into a crouched, chattering
       drift. Train/eval disagreement of that size is nearly always the
       harness, and it was.
    2. The XL330's ~1.75 A current saturation as a symmetric torque clamp
       (kt * I). Training's BAM actuator saturates; plain MuJoCo position
       actuators do not, and unrealistically strong motors change the
       closed-loop behaviour the policy meets.
    """
    model.opt.timestep = 0.005
    if current_limit and current_limit > 0:
        try:
            from bam.model import load_model
            kt = load_model(motor_name="xl330", model="m6").kt.value
        except Exception:
            kt = XL330_KT
        limit = kt * current_limit
        model.actuator_forcerange[:, 0] = -limit
        model.actuator_forcerange[:, 1] = limit
        model.actuator_forcelimited[:] = 1


def rollout(
    policy_onnx: str,
    microduck_rl: str | None,
    controller: BeatController,
    seconds: float = 20.0,
    scene: str | None = None,
    action_scale: float = 1.0,
) -> "object":
    """Step the policy headlessly and return a Trajectory."""
    import mujoco

    infer = load_infer_policy(microduck_rl)

    # Same resolution order load_infer_policy uses, so the scene and the script
    # can never come from different checkouts.
    root_str = microduck_rl or os.environ.get("MICRODUCK_RL")
    if not root_str:
        raise SystemExit("Set MICRODUCK_RL or pass --microduck-rl.")
    root = Path(root_str)
    xml = Path(scene) if scene else root / MICRODUCK_SCENE
    if not xml.exists():
        raise SystemExit(f"scene not found: {xml}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    condition_model(model)
    data = mujoco.MjData(model)
    # Start from the STAND keyframe, found BY NAME. Keyframe 0 in scene.xml is
    # "INIT" — every joint at zero, legs straight — a pose this policy never
    # saw and cannot recover from (fall recovery is a different task). One
    # `mj_resetDataKeyframe(model, data, 0)` quietly turned every eval into
    # "dropped from an alien pose, graded on the aftermath".
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    control_dt = DECIMATION * model.opt.timestep
    recorder = TrajectoryRecorder(
        bpm=controller.bpm,
        dt=control_dt,
        bob=controller.source.bob,
        sway=controller.source.sway,
    )

    cls = make_dance_inference(infer.PolicyInference, controller, recorder)
    policy = cls(
        model,
        data,
        walking_onnx_path=policy_onnx,
        action_scale=action_scale,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )

    for _ in range(int(seconds / control_dt)):
        # Writes the beat clock into the command and records this frame.
        policy.update_ground_pick_phase(control_dt)
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

    return recorder.to_trajectory()

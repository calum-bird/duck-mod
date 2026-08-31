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
    data = mujoco.MjData(model)
    # Start from the STAND keyframe when the scene defines one — otherwise the
    # duck begins collapsed at the origin and the first second of every rollout
    # measures it falling over rather than dancing.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
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

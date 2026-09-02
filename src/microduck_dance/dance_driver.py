"""Drive a trained dance policy in the MuJoCo viewer, and record what it did.

Upstream's ``scripts/infer_policy.py`` already does the hard parts — loading the
ONNX, building the 48-dim proprioception, applying actions at 50 Hz, running the
viewer. It also happens to expose exactly the seam this task needs: its main
loop calls ``policy.update_ground_pick_phase(actual_dt)`` once per frame, and
that method's whole job is to write a phase into the twist slots of the 13-wide
command. A beat clock is the same shape of thing.

So rather than fork 1383 lines, we subclass ``PolicyInference``, override that
one method, and let everything else run untouched. The subclass is installed by
rebinding the module global before calling ``main()`` — ``main`` looks the class
up by name, so the swap is complete and reversible.

Nothing here imports mujoco or the upstream script at module load, so the rest
of the package (and its tests) stay importable on a machine with only torch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from microduck_dance.beat_source import BeatCommandSource

# The two foot collision geoms, from robot_walk.xml. Also the geoms the
# training-time `feet_ground_contact` sensor watches, so eval and reward agree
# on what "a foot is down" means.
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def load_infer_policy(path: str | None = None):
    """Import upstream's ``scripts/infer_policy.py`` as a module.

    It is a script, not an installed package, so it is loaded by file path.
    Point ``MICRODUCK_RL`` at a microduck_rl checkout, or pass ``path``.
    """
    root = path or os.environ.get("MICRODUCK_RL")
    if not root:
        raise SystemExit(
            "Set MICRODUCK_RL to a microduck_rl checkout (it holds "
            "scripts/infer_policy.py), or pass --microduck-rl."
        )
    script = Path(root)
    if script.is_dir():
        script = script / "scripts" / "infer_policy.py"
    if not script.exists():
        raise SystemExit(f"No infer_policy.py at {script}")

    spec = importlib.util.spec_from_file_location("microduck_infer_policy", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # infer_policy resolves sibling imports relative to the repo root.
    sys.path.insert(0, str(script.parent.parent))
    spec.loader.exec_module(module)
    return module


class BeatController:
    """The clock the viewer runs on: fixed tempo, a ramp, or a live file.

    The ramp exists because tempo generalisation is the last thing the
    curriculum teaches and the first thing you want to see fail — sweeping
    60 -> 160 BPM across one episode shows you where the policy lets go, in one
    recording, instead of across a dozen fixed-tempo runs.
    """

    def __init__(
        self,
        bpm: float = 120.0,
        bob: float = 0.018,
        sway: float = 0.014,
        yaw: float = 0.15,
        sweep_to: float | None = None,
        sweep_seconds: float = 60.0,
        bpm_file: str | None = None,
        on_start=None,
    ):
        self.source = BeatCommandSource(bpm=bpm, bob=bob, sway=sway, yaw=yaw)
        self._bpm_start = bpm
        self._sweep_to = sweep_to
        self._sweep_seconds = max(sweep_seconds, 1e-6)
        self._bpm_file = bpm_file
        self._elapsed = 0.0
        # Fires on the FIRST advance(), i.e. when the sim loop actually starts
        # stepping — not at construction, which is separated from the first
        # frame by seconds of model loading. This is the only moment that can
        # start external audio in time with the clock's beat 1.
        self._on_start = on_start

    @property
    def bpm(self) -> float:
        return self.source.bpm

    @property
    def phase(self) -> float:
        return self.source.phase

    def _poll_file(self) -> None:
        """Live tempo control without touching upstream's key handling.

        Deliberately a file rather than a keybinding: it works over SSH, from a
        second shell, and from a tap-tempo script, and it needs no changes to
        the viewer we are borrowing.
        """
        if not self._bpm_file:
            return
        try:
            text = Path(self._bpm_file).read_text().strip()
        except (OSError, ValueError):
            return
        try:
            self.source.set_tempo(float(text))
        except ValueError:
            pass

    def advance(self, dt: float) -> list[float]:
        if self._on_start is not None:
            cb, self._on_start = self._on_start, None
            cb()
        self._elapsed += dt
        if self._sweep_to is not None:
            frac = min(self._elapsed / self._sweep_seconds, 1.0)
            self.source.set_tempo(
                self._bpm_start + (self._sweep_to - self._bpm_start) * frac
            )
        self._poll_file()
        return self.source.tick(dt)


class FootContactReader:
    """Per-foot ground contact straight from MuJoCo's contact list.

    Read from the physics rather than from the policy's observation so the
    measurement stays honest even if the observation is what is broken.
    """

    def __init__(self, model):
        import mujoco

        self._ids = []
        for name in FOOT_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise SystemExit(f"Geom '{name}' not in the model — wrong XML?")
            self._ids.append(gid)

    def read(self, data) -> tuple[bool, bool]:
        down = [False, False]
        for i in range(data.ncon):
            c = data.contact[i]
            for slot, gid in enumerate(self._ids):
                if c.geom1 == gid or c.geom2 == gid:
                    down[slot] = True
        return down[0], down[1]


class TrajectoryRecorder:
    """Accumulate the per-frame state the eval metrics need."""

    def __init__(self, bpm: float, dt: float, bob: float, sway: float):
        self.bpm, self.dt, self.bob, self.sway = bpm, dt, bob, sway
        self.bar_phase: list[float] = []
        self.contacts: list[tuple[bool, bool]] = []
        self.root_z: list[float] = []
        self.root_xy: list[tuple[float, float]] = []
        self.root_yaw: list[float] = []
        self.head_yaw: list[float] = []

    def record(self, phase: float, data, contacts: tuple[bool, bool],
               head_yaw: float = 0.0) -> None:
        import math

        # A free-joint root: qpos is [x, y, z, qw, qx, qy, qz, ...]. This is
        # MuJoCo's own layout, not an mjlab convention, so it holds for any of
        # the microduck XMLs.
        x, y, z = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
        qw, qx, qy, qz = (float(data.qpos[i]) for i in range(3, 7))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        self.bar_phase.append(phase)
        self.contacts.append(contacts)
        self.root_z.append(z)
        self.root_xy.append((x, y))
        self.root_yaw.append(yaw)
        self.head_yaw.append(head_yaw)

    def to_trajectory(self):
        import torch

        from microduck_dance.eval import Trajectory

        return Trajectory(
            bpm=self.bpm,
            dt=self.dt,
            bar_phase=torch.tensor(self.bar_phase),
            contacts=torch.tensor(self.contacts, dtype=torch.bool),
            root_z=torch.tensor(self.root_z),
            root_xy=torch.tensor(self.root_xy),
            root_yaw=torch.tensor(self.root_yaw),
            head_yaw=torch.tensor(self.head_yaw),
            commanded_bob=self.bob,
            commanded_sway=self.sway,
        )


def make_dance_inference(base_cls, controller: BeatController, recorder=None):
    """Subclass upstream's PolicyInference with a beat clock in the twist slot."""

    class DanceInference(base_cls):  # type: ignore[valid-type,misc]
        beat = controller
        _recorder = recorder
        _contact_reader = None
        _head_adr = None

        def update_ground_pick_phase(self, dt: float) -> None:
            """Called once per viewer frame with the real dt (upstream loop).

            Writes the beat clock into twist[0:3] and the choreography
            amplitudes into the body_pose slots, leaving head_pose[3:7] to
            upstream's existing keyboard mapping so the head stays steerable.
            Runs after key handling in the frame, so a keypress that rewrites
            the whole command cannot clobber the clock.
            """
            cmd = self.beat.advance(dt)
            self.command[0:3] = cmd[0:3]
            self.command[7:13] = cmd[7:13]

            if self._recorder is not None:
                import mujoco

                if self._contact_reader is None:
                    self._contact_reader = FootContactReader(self.model)
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
                    type(self)._head_adr = int(self.model.jnt_qposadr[jid]) if jid >= 0 else None
                head = float(self.data.qpos[self._head_adr]) if self._head_adr is not None else 0.0
                self._recorder.record(
                    self.beat.phase, self.data,
                    self._contact_reader.read(self.data), head_yaw=head,
                )

    return DanceInference

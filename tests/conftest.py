"""A stub environment just rich enough to exercise the dance reward terms.

The reward functions touch a deliberately narrow slice of the mjlab env API --
root pose, the foot contact sensor, the command manager and a handful of
counters -- which is what makes them testable on a CPU box with no simulator.
Anything that needs the real physics (joint-id resolution, MJCF parsing, the
BAM actuator) lives in the config-level tests instead, which skip without mjlab.
"""

from __future__ import annotations

import math

import pytest
import torch


class _Data:
    def __init__(self, pos, quat):
        self.root_link_pos_w = pos
        self.root_link_quat_w = quat
        self.root_link_lin_vel_w = torch.zeros_like(pos)
        self.joint_pos = torch.zeros(pos.shape[0], 14)
        self.joint_vel = torch.zeros(pos.shape[0], 14)


class _Asset:
    def __init__(self, pos, quat):
        self.data = _Data(pos, quat)


class _Sensor:
    def __init__(self, found):
        self.data = type("D", (), {"found": found})()


class _Terrain:
    def __init__(self, origins):
        self.env_origins = origins


class _Scene:
    def __init__(self, asset, origins, sensors):
        self._asset = asset
        self.terrain = _Terrain(origins)
        self.sensors = sensors

    def __getitem__(self, name):
        return self._asset


class _CommandManager:
    def __init__(self, commands, terms=None):
        self._commands = commands
        self._terms = terms or {}

    def get_command(self, name):
        return self._commands[name]

    def get_term(self, name):
        return self._terms.get(name)


class StubEnv:
    """Minimal stand-in for ManagerBasedRlEnv."""

    def __init__(
        self,
        num_envs: int = 4,
        height: float = 0.095,
        yaw: float = 0.0,
        xy=None,
        contacts=None,
        commands=None,
        step_dt: float = 0.02,
        common_step_counter: int = 0,
    ):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = step_dt
        self.common_step_counter = common_step_counter
        # >1 so buffers are not treated as freshly reset unless a test says so.
        self.episode_length_buf = torch.full((num_envs,), 10, dtype=torch.long)

        xy = torch.zeros(num_envs, 2) if xy is None else xy
        pos = torch.zeros(num_envs, 3)
        pos[:, :2] = xy
        pos[:, 2] = height
        self.set_yaw(yaw, pos)

        contacts = (
            torch.zeros(num_envs, 2, dtype=torch.bool) if contacts is None else contacts
        )
        self.scene = _Scene(
            self._asset,
            torch.zeros(num_envs, 3),
            {"feet_ground_contact": _Sensor(contacts)},
        )
        self.command_manager = _CommandManager(commands or {})
        # Neck/head joint cache in command order, as upstream resolves it:
        # [neck_pitch, head_pitch, head_yaw, head_roll]
        self._head_pose_neck_ids = [5, 6, 7, 8]

    def set_yaw(self, yaw, pos=None):
        if pos is None:
            pos = self._asset.data.root_link_pos_w
        n = self.num_envs
        yaw_t = torch.as_tensor(yaw, dtype=torch.float32).expand(n).clone()
        quat = torch.zeros(n, 4)
        quat[:, 0] = torch.cos(yaw_t / 2)
        quat[:, 3] = torch.sin(yaw_t / 2)
        old = getattr(self, "_asset", None)
        self._asset = _Asset(pos, quat)
        if old is not None:
            self._asset.data.root_link_lin_vel_w = old.data.root_link_lin_vel_w
            self._asset.data.joint_pos = old.data.joint_pos
            self._asset.data.joint_vel = old.data.joint_vel
        if hasattr(self, "scene"):
            self.scene._asset = self._asset

    def set_tilt(self, tilt_deg: float):
        """Tilt about x, which is what the gate's cos_tilt term measures."""
        n = self.num_envs
        half = math.radians(tilt_deg) / 2
        quat = torch.zeros(n, 4)
        quat[:, 0] = math.cos(half)
        quat[:, 1] = math.sin(half)
        self._asset.data.root_link_quat_w = quat

    def set_height(self, z: float):
        self._asset.data.root_link_pos_w[:, 2] = z

    def set_xy(self, xy: torch.Tensor):
        self._asset.data.root_link_pos_w[:, :2] = xy

    def set_joint_vel(self, idx: int, value: float):
        self._asset.data.joint_vel[:, idx] = value

    def set_lin_vel(self, vel: torch.Tensor):
        self._asset.data.root_link_lin_vel_w[:] = vel

    def set_contacts(self, contacts: torch.Tensor):
        self.scene.sensors["feet_ground_contact"].data.found = contacts

    def set_commands(self, **commands):
        self.command_manager._commands.update(commands)


def twist_at(phase: float, num_envs: int = 4, tempo_norm: float = 0.0) -> torch.Tensor:
    """Beat-clock command for a given bar phase."""
    cmd = torch.zeros(num_envs, 3)
    cmd[:, 0] = math.cos(2 * math.pi * phase)
    cmd[:, 1] = math.sin(2 * math.pi * phase)
    cmd[:, 2] = tempo_norm
    return cmd


def amplitudes(bob=0.02, sway=0.015, yaw=0.15, num_envs: int = 4) -> torch.Tensor:
    """body_pose slot carrying choreography amplitudes."""
    cmd = torch.zeros(num_envs, 6)
    cmd[:, 1] = sway
    cmd[:, 2] = bob
    cmd[:, 5] = yaw
    return cmd


@pytest.fixture
def env():
    e = StubEnv()
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes())
    return e

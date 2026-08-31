"""Reward, penalty and curriculum terms for the beat-synced dance task.

Sign convention (from the upstream reward playbook, ``AGENTS.md``):

* tracking rewards return values in ``[0, 1]`` and take **positive** weights;
* penalties are **self-negating** -- they return ``<= 0`` and also take
  positive weights, so that every ``Episode_Reward/<name>_penalty`` series in
  wandb is ``<= 0``. A positive weight on a cost-style (``>= 0``) return is the
  double-negative bug that makes a policy farm violations.

Every positive term here is multiplied by :func:`dance_gate`, a hard state gate
(upright, above a height floor). This is deliberate and follows the playbook's
"the letter, not the spirit" warning: a rhythmic reward that pays while the
robot is down is farmable by sitting on the trunk and bouncing to the beat,
which is both a plausible local optimum and a bad time on real hardware. Soft
nudge penalties do not close that door; a hard gate does.

These functions take ``asset_name``/``sensor_name`` strings rather than
``SceneEntityCfg`` objects. They only ever need the root body and the foot
contact sensor -- no joint-id resolution -- and the plain-string signature
keeps this module importable (and therefore unit-testable) without mjlab.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from microduck_dance import beat_clock

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# --------------------------------------------------------------------------- #
# State helpers                                                               #
# --------------------------------------------------------------------------- #


def _root_state(env, asset_name: str):
    asset = env.scene[asset_name]
    origins = env.scene.terrain.env_origins
    pos = torch.nan_to_num(asset.data.root_link_pos_w - origins, nan=0.0)
    quat = asset.data.root_link_quat_w  # (w, x, y, z)
    return pos, quat


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _body_y_axis(quat: torch.Tensor) -> torch.Tensor:
    """World-frame y axis of the trunk, i.e. the robot's own 'left'."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z)], dim=-1)


def dance_gate(
    env,
    asset_name: str = "robot",
    min_height: float = 0.070,
    max_tilt_deg: float = 25.0,
) -> torch.Tensor:
    """Hard 0/1 mask: 1 only while standing tall and near-upright.

    Height floor is below the nominal 0.095 m stance so a deep, legitimate bob
    still counts, but well above any posture that rests on the trunk.
    """
    pos, quat = _root_state(env, asset_name)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = cos_tilt >= math.cos(math.radians(max_tilt_deg))
    tall = pos[:, 2] >= min_height
    return (upright & tall).float()


def _amplitudes(env, body_command_name: str, max_bob: float, max_sway: float, max_yaw: float):
    """Read the choreography amplitudes out of the body_pose command slot.

    The 6-wide body_pose slot nominally means a body offset in
    ``[x, y, z, roll, pitch, yaw]``. This task reinterprets three of those dims
    as the *peak amplitude* of the rhythmic motion along the very same axis --
    y -> sway, z -> bob, yaw -> twist -- so the slot keeps its axis semantics
    and the runtime keeps its 61-dim buffer. x/roll/pitch stay zero-padded.
    """
    cmd = env.command_manager.get_command(body_command_name)
    sway = cmd[:, 1].clamp(0.0, max_sway)
    bob = cmd[:, 2].clamp(0.0, max_bob)
    yaw = cmd[:, 5].clamp(0.0, max_yaw)
    return bob, sway, yaw


# --------------------------------------------------------------------------- #
# Choreography tracking rewards                                               #
# --------------------------------------------------------------------------- #


def beat_bob_tracking(
    env,
    command_name: str = "twist",
    body_command_name: str = "body_pose",
    nominal_height: float = 0.095,
    std: float = 0.010,
    max_bob: float = 0.025,
    max_sway: float = 0.020,
    max_yaw: float = 0.25,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Gaussian reward for the trunk height following the on-beat bob.

    ``std`` is set to the height error we still care about (1 cm on a 2.5 cm
    peak bob), not to the worst case: too loose and the gradient dies exactly
    where the timing gets refined; too tight and the term becomes an
    inescapable tax on a robot whose head is 38% of its mass.
    """
    cmd = env.command_manager.get_command(command_name)
    phase = beat_clock.bar_phase_from_command(cmd)
    bob, _, _ = _amplitudes(env, body_command_name, max_bob, max_sway, max_yaw)
    z_ref = beat_clock.bob_reference(phase, bob, nominal_height)
    pos, _ = _root_state(env, asset_name)
    reward = torch.exp(-(((pos[:, 2] - z_ref) / std) ** 2))
    return reward * dance_gate(env, asset_name)


def beat_sway_tracking(
    env,
    command_name: str = "twist",
    body_command_name: str = "body_pose",
    std: float = 0.012,
    max_bob: float = 0.025,
    max_sway: float = 0.020,
    max_yaw: float = 0.25,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Gaussian reward for the lateral weight shift following the bar sway.

    Offset is measured along the trunk's own y axis, so the term means "sway"
    and not "drift east" -- otherwise a duck that had yawed 90 degrees would be
    scored on the wrong axis.
    """
    cmd = env.command_manager.get_command(command_name)
    phase = beat_clock.bar_phase_from_command(cmd)
    _, sway, _ = _amplitudes(env, body_command_name, max_bob, max_sway, max_yaw)
    y_ref = beat_clock.sway_reference(phase, sway)
    pos, quat = _root_state(env, asset_name)
    lateral = (pos[:, :2] * _body_y_axis(quat)).sum(dim=-1)
    reward = torch.exp(-(((lateral - y_ref) / std) ** 2))
    return reward * dance_gate(env, asset_name)


def beat_yaw_tracking(
    env,
    command_name: str = "twist",
    body_command_name: str = "body_pose",
    std: float = 0.15,
    max_bob: float = 0.025,
    max_sway: float = 0.020,
    max_yaw: float = 0.25,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Gaussian reward for the trunk twisting into the sway (same bar period)."""
    cmd = env.command_manager.get_command(command_name)
    phase = beat_clock.bar_phase_from_command(cmd)
    _, _, yaw_amp = _amplitudes(env, body_command_name, max_bob, max_sway, max_yaw)
    yaw_ref = beat_clock.yaw_reference(phase, yaw_amp)
    _, quat = _root_state(env, asset_name)
    err = torch.remainder(_yaw_from_quat(quat) - yaw_ref + math.pi, 2 * math.pi) - math.pi
    reward = torch.exp(-((err / std) ** 2))
    return reward * dance_gate(env, asset_name)


# --------------------------------------------------------------------------- #
# Footfall timing                                                             #
# --------------------------------------------------------------------------- #


def _foot_contacts(env, sensor_name: str) -> torch.Tensor:
    """Per-foot boolean contact, shape (N, 2), left/right in sensor order."""
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() == 1:
        found = found.unsqueeze(-1)
    return found > 0


def beat_footfall_reward(
    env,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    sigma: float = beat_clock.DOWNBEAT_SIGMA,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Pay a foot strike once per beat, scaled by how close it was to the beat.

    Paid on the *rising edge* of contact, and rate-limited to one payout per
    beat. Both properties matter. A per-step "a foot is down near the beat"
    reward is the playbook's jackpot pattern: the policy would plant a foot on
    the downbeat and hold it there, collecting for the whole beat while
    standing still. Here only arriving pays, and arriving twice for the same
    beat pays once.

    The rate limit is keyed to the NEAREST downbeat, not to the beat currently
    in progress. A strike at beat-phase 0.98 is an early strike for the *next*
    downbeat, so it must consume that downbeat's payout -- otherwise a policy
    that taps just before the beat and again just after collects twice for one
    beat, which is precisely the double-dip a naive wrap-triggered re-arm
    allows.
    """
    cmd = env.command_manager.get_command(command_name)
    phase = beat_clock.bar_phase_from_command(cmd)
    contacts = _foot_contacts(env, sensor_name)

    if not hasattr(env, "_dance_prev_contact"):
        env._dance_prev_contact = torch.zeros_like(contacts)
        env._dance_prev_beat_phase = torch.zeros(env.num_envs, device=env.device)
        env._dance_beat_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._dance_last_paid = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    if bool(fresh.any()):
        env._dance_prev_contact[fresh] = False
        env._dance_prev_beat_phase[fresh] = 0.0
        env._dance_beat_count[fresh] = 0
        env._dance_last_paid[fresh] = -1

    b = beat_clock.beat_phase(phase)
    # Phase decreasing means the beat wrapped; count beats monotonically so the
    # payout account survives the wrap.
    wrapped = b < env._dance_prev_beat_phase
    env._dance_beat_count = env._dance_beat_count + wrapped.long()
    env._dance_prev_beat_phase = b
    # Index of the downbeat this instant belongs to: the current one in the
    # first half of the beat, the upcoming one in the second half.
    nearest_beat = env._dance_beat_count + (b >= 0.5).long()

    landed = (contacts & ~env._dance_prev_contact).any(dim=-1)
    env._dance_prev_contact = contacts.clone()

    payable = landed & (nearest_beat != env._dance_last_paid)
    env._dance_last_paid = torch.where(payable, nearest_beat, env._dance_last_paid)

    proximity = beat_clock.downbeat_proximity(phase, sigma)
    return proximity * payable.float() * dance_gate(env, asset_name)


def foot_alternation_penalty(
    env,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    asset_name: str = "robot",
) -> torch.Tensor:
    """Charge a strike that repeats the previous strike's foot. Returns <= 0.

    Without it the cheapest way to satisfy :func:`beat_footfall_reward` is to
    tap the same foot every beat while the other stays planted -- on beat, but
    not a dance. 'Both feet at once' is tracked as its own id so landing two
    feet every beat is a repeat too, rather than a loophole.
    """
    contacts = _foot_contacts(env, sensor_name)

    if not hasattr(env, "_dance_last_foot"):
        env._dance_last_foot = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        env._dance_alt_prev_contact = torch.zeros_like(contacts)
    fresh = env.episode_length_buf <= 1
    if bool(fresh.any()):
        env._dance_last_foot[fresh] = -1
        env._dance_alt_prev_contact[fresh] = False

    new = contacts & ~env._dance_alt_prev_contact
    env._dance_alt_prev_contact = contacts.clone()

    left = new[:, 0]
    right = new[:, 1] if new.shape[-1] > 1 else new[:, 0]
    both = left & right
    # 0 = left, 1 = right, 2 = both, -1 = nothing landed this step.
    foot_id = torch.where(
        both,
        torch.full_like(env._dance_last_foot, 2),
        torch.where(
            left,
            torch.zeros_like(env._dance_last_foot),
            torch.where(
                right,
                torch.ones_like(env._dance_last_foot),
                torch.full_like(env._dance_last_foot, -1),
            ),
        ),
    )
    landed = foot_id >= 0
    repeat = landed & (foot_id == env._dance_last_foot)
    env._dance_last_foot = torch.where(landed, foot_id, env._dance_last_foot)
    return -repeat.float()


# --------------------------------------------------------------------------- #
# Station keeping                                                             #
# --------------------------------------------------------------------------- #


def station_keeping_penalty(
    env,
    tau_s: float = 4.0,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Charge the time-averaged horizontal drift: ``-|EMA(xy offset)|``. <= 0.

    The EMA is the whole point. An instantaneous |xy| penalty would fight the
    sway reward directly -- the sway *is* horizontal displacement -- and the two
    would net out to a duck that stands rigidly still. Averaging over a window
    longer than the slowest bar (2.0 s at 60 BPM) lets the oscillation cancel
    and prices only the DC component, i.e. actually wandering off the spot.
    This is the same escapable-bias-only trick the upstream head_pose_bias
    penalty uses against head droop.
    """
    pos, _ = _root_state(env, asset_name)
    xy = pos[:, :2]
    if not hasattr(env, "_dance_xy_ema"):
        env._dance_xy_ema = torch.zeros_like(xy)
    fresh = env.episode_length_buf <= 1
    env._dance_xy_ema[fresh] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._dance_xy_ema = (1.0 - alpha) * env._dance_xy_ema + alpha * xy
    return -torch.norm(env._dance_xy_ema, dim=-1)


def heading_drift_penalty(
    env,
    tau_s: float = 4.0,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Charge the time-averaged yaw offset: ``-|EMA(yaw)|``. <= 0.

    Same reasoning as :func:`station_keeping_penalty`: the choreographed twist
    oscillates around zero and cancels in the average, so only a duck that
    slowly rotates off its mark pays.
    """
    _, quat = _root_state(env, asset_name)
    yaw = _yaw_from_quat(quat)
    if not hasattr(env, "_dance_yaw_ema"):
        env._dance_yaw_ema = torch.zeros_like(yaw)
    fresh = env.episode_length_buf <= 1
    env._dance_yaw_ema[fresh] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._dance_yaw_ema = (1.0 - alpha) * env._dance_yaw_ema + alpha * yaw
    return -env._dance_yaw_ema.abs()


# --------------------------------------------------------------------------- #
# Curriculum                                                                  #
# --------------------------------------------------------------------------- #


def beat_tempo_range_curriculum(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Widen the sampled BPM envelope over training.

    Mutates the live CommandManager term cfg, never ``env.cfg`` -- the managers
    deepcopy their config at init, so writing to ``env.cfg`` is a silent no-op.
    ``range_stages`` is a list of ``{"step": int, "bpm_range": (lo, hi)}``.
    """
    del env_ids
    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"

    current = range_stages[0]["bpm_range"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["bpm_range"]
    term.cfg.bpm_range = tuple(current)
    return torch.tensor(float(current[1] - current[0]))


def beat_tempo_change_prob_curriculum(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    prob_stages: list[dict],
) -> torch.Tensor:
    """Ramp the probability of a mid-episode tempo change.

    Held at zero until the policy can hold a steady tempo: re-locking to a new
    BPM is a strictly harder problem than tracking one, and introducing it
    early just adds variance to the signal the policy is still trying to find.
    ``prob_stages`` is a list of ``{"step": int, "prob": float}``.
    """
    del env_ids
    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"

    current = prob_stages[0]["prob"]
    for stage in prob_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["prob"]
    term.cfg.tempo_change_prob = float(current)
    return torch.tensor(float(current))

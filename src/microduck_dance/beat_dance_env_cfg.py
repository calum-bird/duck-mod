"""Microduck beat-synced dance environment.

Teaches the duck to dance *in place* to an external beat: bob on every beat,
shift its weight left/right across each bar, twist into the sway, and land a
foot on the downbeat -- at whatever tempo the host is sending, without
wandering off its spot.

Built as a derivation of the shipped velocity env (same robot, same 61-dim
observation contract, same domain randomisation, same actuator model), with
three changes:

1.  The ``twist`` slot stops being a velocity command and becomes a beat clock
    (see ``beat_clock``). This is the established pattern for a non-locomotion
    microduck task -- ground-pick, sit-stand and spin all repurpose the same
    slot -- and it is what keeps the observation vector at 61 dims so the
    runtime can hot-swap this policy alongside the walking one.
2.  The ``body_pose`` slot stops being a pose target and becomes the
    choreography amplitudes (y -> sway, z -> bob, yaw -> twist), so the host
    can dial the dance from subtle to exuberant at run time, zero included.
3.  The walking reward recipe is swapped for a rhythm recipe: velocity
    tracking and air-time come out, phase-tracking and on-beat footfall go in.

The curriculum below is phase-aligned rather than simultaneous: the duck learns
a stable bob first, then the weight shift, then on-beat foot placement, and
only then faces a wide tempo range. Introducing the footfall term at step 0
would ask it to place a foot on a beat before it can stand rhythmically at all.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

from microduck_dance import mdp as dance_mdp
from microduck_dance.beat_clock import BPM_MAX, BPM_MIN, NOMINAL_HEIGHT
from microduck_dance.commands import BeatClockCommand, BeatClockCommandCfg

# Rollout length used upstream; curriculum steps are iterations x this.
NUM_STEPS_PER_ENV = 24

# Curriculum stage boundaries, in PPO iterations.
# Run 1 introduced sway at 800 while the bob had not consolidated (it was flat
# at 0.25 and the duck was still falling), which is exactly the pacing error the
# playbook warns about: do not harden a stage before the current one holds.
_STAGE_SWAY = 1200
_STAGE_FOOTFALL = 2000
_STAGE_ALTERNATION = 2400
_STAGE_TEMPO_WIDE = 3000
_STAGE_FULL = 3800

# Choreography envelopes. The duck stands ~0.095 m at the trunk, so a 25 mm bob
# is a deep one and 20 mm of sway is a full weight commit.
MAX_BOB = 0.025
MAX_SWAY = 0.020
MAX_YAW = 0.25

# Reward MASS of the footfall impulse, not its weight. It pays at most once per
# beat: at 120 BPM that is 2 payouts/s against a 50 Hz control loop, so only 4%
# of steps pay anything. A weight of 40 with a typical proximity of ~0.7 is
# therefore worth 40 * 0.04 * 0.7 ~= 1.1 per step on average -- the same order
# as the bob term's 4.0 * ~0.8 = 3.2, which is the comparison that matters to
# PPO. Weighting it like a dense term (say 4.0) would make it worth 0.1/step
# and the policy would simply ignore the beat.
FOOTFALL_WEIGHT = 40.0
ALTERNATION_WEIGHT = 20.0

# Walking terms that either read the twist slot as a velocity (and so are
# meaningless once it carries a phase) or actively fight a prescribed rhythm.
_WALKING_TERMS = (
    "track_linear_velocity",   # twist is no longer a velocity
    "track_angular_velocity",  # idem
    "air_time",                # rewards LONG swings; the beat sets the cadence
)


def make_microduck_beat_dance_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ── Command slot 1: twist -> beat clock ──────────────────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # Carry over only the fields BeatClockCommandCfg actually declares. The
    # velocity env swaps in its own cfg subclass carrying extra knobs
    # (rel_turn_in_place_envs, and whatever it grows next), and splatting those
    # into a different dataclass is a TypeError at import — which surfaces as
    # the whole task package silently failing to register.
    _allowed = {f.name for f in dataclasses.fields(BeatClockCommandCfg)}
    _carried = {k: v for k, v in vars(command).items() if k in _allowed}
    cfg.commands["twist"] = BeatClockCommandCfg(
        **{
            **_carried,
            "class_type": BeatClockCommand,
            # Start at an effectively fixed 120 BPM: one tempo to learn before
            # being asked to generalise across tempi. Widened by curriculum.
            "bpm_range": (118.0, 122.0),
            # In play mode every episode starts on a downbeat, which is what
            # the runtime does when the host starts the music.
            "randomize_phase": not play,
            "tempo_change_prob": 0.0,  # raised at _STAGE_FULL
            # Resample tick only drives mid-episode tempo changes; the phase is
            # never resampled.
            "resampling_time_range": (4.0, 8.0),
        }
    )

    # ── Command slot 3: body_pose -> choreography amplitudes ─────────────────
    # Amplitudes are non-negative by construction (a negative "amplitude" is
    # just a half-bar phase shift, which the clock already covers). Stage 1
    # pins them near the middle of the envelope so the reference the policy is
    # chasing is unambiguous while it is still learning to chase anything.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(4.0, 8.0),
        ranges=(
            (0.0, 0.0),      # x  — unused, zero-padded
            (0.010, 0.014),  # y  — sway amplitude (m)
            (0.012, 0.016),  # z  — bob amplitude (m)
            (0.0, 0.0),      # roll  — unused
            (0.0, 0.0),      # pitch — unused
            (0.10, 0.14),    # yaw — twist amplitude (rad)
        ),
    )

    # ── Rewards: out with the walking recipe ─────────────────────────────────
    for name in _WALKING_TERMS:
        if name in cfg.rewards:
            del cfg.rewards[name]
    # body_pose_tracking would read the amplitude slot as a pose target and
    # pull the trunk toward a 12 mm permanent offset. Delete rather than
    # zero-weight it: at weight 0 it is a loaded footgun for the next reader.
    cfg.rewards.pop("body_pose_tracking", None)
    # The head must be free to oscillate -- it is 38% of the robot's mass and
    # the nod is half of what makes this read as dancing. Upstream's own
    # finding is that an instantaneous head-tracking tax kills stepping
    # outright, so keep only the DC term: the commanded attitude is enforced on
    # a 1 s average, and the bob rides on top of it for free.
    if "head_pose_tracking" in cfg.rewards:
        cfg.rewards["head_pose_tracking"].weight = 0.0
    if "head_pose_bias" in cfg.rewards:
        cfg.rewards["head_pose_bias"].weight = 1.0

    # `pose` was deleted outright in run 1 on the reasoning that a HOME-pull
    # fights the bob. It does -- but it was also the only thing holding the leg
    # pitch chain in a sane stance, and without it the duck spent ~47% of every
    # episode below the gate height. A bob of +-14 mm is a few degrees of knee
    # and ankle, which a WEAK pull tolerates. So keep it, at 40% weight and
    # scoped to the legs (neck and head must stay free to oscillate).
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].weight *= 0.4
        for _std_key in ("std_standing", "std_walking", "std_running"):
            if _std_key in cfg.rewards["pose"].params:
                _d = cfg.rewards["pose"].params[_std_key]
                cfg.rewards["pose"].params[_std_key] = {
                    k: v
                    for k, v in _d.items()
                    if "neck" not in k and "head" not in k and "passive" not in k
                }
        cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
        )

    # angular_momentum penalises the full 3D momentum norm, so it opposes the
    # commanded yaw twist. Kept (it damps genuine flailing) but demoted, per
    # the playbook's rule that motion blockers stay low-weight on dynamic tasks.
    if "angular_momentum" in cfg.rewards:
        cfg.rewards["angular_momentum"].weight *= 0.25
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 2.0

    # ── Rewards: in with the rhythm recipe ───────────────────────────────────
    _amp_params = {
        "max_bob": MAX_BOB,
        "max_sway": MAX_SWAY,
        "max_yaw": MAX_YAW,
    }
    # === The move -> in-time -> exactly ordering (probe-derived, post run 1) ===
    # Every position Gaussian is partly satisfiable by standing at the
    # reference's mean, and the hard gate taxes any motion that risks a fall.
    # scripts/reward_probe.py shows that with the Gaussians at full weight from
    # step 0, a clumsy first attempt at dancing earns LESS than a statue at
    # every std tried (-0.96/step at run 1's config) -- a local optimum with no
    # exit, which is exactly what run 1 trained. With precision ramped in later
    # over motion terms that pay stillness EXACTLY zero, the probe gives
    # +0.51/step out of stillness at stage 1 and +2.65/step against reverting
    # at stage 3, while still retrodicting run 1's failure.
    #
    # beat_bob_power: signed correlation of vertical velocity with the
    # reference's derivative. Stillness earns zero, the first in-phase wobble
    # pays, anti-phase motion is charged. Full weight from step 0.
    cfg.rewards["beat_bob_power"] = RewardTermCfg(
        func=dance_mdp.beat_bob_power,
        weight=dance_mdp.POWER_WEIGHT,
        params={"command_name": "twist", "body_command_name": "body_pose", **_amp_params},
    )
    # beat_bob_energy: phase-BLIND vertical motion of the commanded scale. An
    # early policy's phase error sits near quadrature, where the correlation
    # pays nothing -- this pays the moment it moves at all, and the curriculum
    # decays it once timing terms take over.
    cfg.rewards["beat_bob_energy"] = RewardTermCfg(
        func=dance_mdp.beat_bob_energy,
        weight=dance_mdp.ENERGY_WEIGHT,
        params={"command_name": "twist", "body_command_name": "body_pose", **_amp_params},
    )
    cfg.rewards["beat_sway_power"] = RewardTermCfg(
        func=dance_mdp.beat_sway_power,
        weight=0.0,  # ramped with the sway curriculum (stage-0 must match)
        params={"command_name": "twist", "body_command_name": "body_pose", **_amp_params},
    )
    cfg.rewards["beat_bob"] = RewardTermCfg(
        func=dance_mdp.beat_bob_tracking,
        weight=1.0,  # PRECISION term: small but alive at step 0, ramped to 4.0
                     # by the beat_bob_weight curriculum once motion exists
        params={
            "command_name": "twist",
            "body_command_name": "body_pose",
            "nominal_height": NOMINAL_HEIGHT,
            # 0.010 -> 0.006. At 0.010 against a 14 mm amplitude, simply HOLDING
            # the nominal height scores 0.471 of maximum -- measured, not
            # guessed -- against 0.914 for tracking well. A 1.94x edge does not
            # pay for the fall risk that bobbing carries, so run 1 (2026-08-31)
            # learned to stand rigidly still: beat_bob sat at 0.25, which is
            # 0.471 x a 53%-open gate almost exactly. At 0.006 the stand-still
            # payoff drops to 0.257 while good tracking still earns 0.779 -- a
            # 3.0x edge. See scripts/bob_reward_payoff.py for the table.
            "std": 0.006,
            **_amp_params,
        },
    )
    cfg.rewards["beat_sway"] = RewardTermCfg(
        func=dance_mdp.beat_sway_tracking,
        weight=0.0,  # ramped at _STAGE_SWAY (must match the curriculum's step-0 value)
        params={
            "command_name": "twist",
            "body_command_name": "body_pose",
            # 0.012 -> 0.007: the same stand-still loophole beat_bob had, one
            # file-width away. At std equal to the 12 mm amplitude, a statue
            # collects 0.645 of this term for free.
            "std": 0.007,
            **_amp_params,
        },
    )
    cfg.rewards["beat_yaw"] = RewardTermCfg(
        func=dance_mdp.beat_yaw_tracking,
        weight=0.0,  # ramped at _STAGE_FULL
        params={
            "command_name": "twist",
            "body_command_name": "body_pose",
            "std": 0.15,
            **_amp_params,
        },
    )
    cfg.rewards["beat_footfall"] = RewardTermCfg(
        func=dance_mdp.beat_footfall_reward,
        weight=0.0,  # ramped at _STAGE_FOOTFALL
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["foot_alternation_penalty"] = RewardTermCfg(
        func=dance_mdp.foot_alternation_penalty,
        weight=0.0,  # ramped at _STAGE_ALTERNATION
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    )
    # Self-negating penalties (<= 0) take POSITIVE weights, per the sign
    # convention: their wandb series must stay <= 0.
    cfg.rewards["station_keeping_penalty"] = RewardTermCfg(
        func=dance_mdp.station_keeping_penalty,
        weight=20.0,  # 10 cm of averaged drift costs 2.0/step
        params={"tau_s": 4.0},
    )
    cfg.rewards["heading_drift_penalty"] = RewardTermCfg(
        func=dance_mdp.heading_drift_penalty,
        weight=4.0,  # 0.5 rad of averaged rotation costs 2.0/step
        params={"tau_s": 4.0},
    )
    # `pose` is gone, so nothing was left holding the stance narrow. This is
    # cost-style (>= 0) and therefore takes a NEGATIVE weight. Scoped to the
    # yaw/roll hips only -- the axes the dance has no business using -- so it
    # never fights the pitch chain doing the bob.
    cfg.rewards["stance_width_penalty"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*")
            )
        },
    )

    # ── Terminations ─────────────────────────────────────────────────────────
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Curriculum ───────────────────────────────────────────────────────────
    # Step functions, not interpolations, and each stage waits for the previous
    # skill to consolidate. Every stage-0 weight must equal the term's initial
    # weight above, or the first curriculum tick silently moves the goalposts.
    def _stage(reward_name: str, stages: list[dict]) -> CurriculumTermCfg:
        return CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": reward_name, "weight_stages": stages},
        )

    # Precision ramps: bob Gaussian grows only as the motion terms hand over.
    cfg.curriculum["beat_bob_weight"] = _stage(
        "beat_bob",
        [
            {"step": 0, "weight": 1.0},
            {"step": 600 * NUM_STEPS_PER_ENV, "weight": 2.5},
            {"step": _STAGE_SWAY * NUM_STEPS_PER_ENV, "weight": 4.0},
        ],
    )
    # The bootstrap decays but keeps a floor: motion must never again earn
    # strictly less than a statue, whatever the Gaussians misprice.
    cfg.curriculum["beat_bob_energy_weight"] = _stage(
        "beat_bob_energy",
        [
            {"step": 0, "weight": dance_mdp.ENERGY_WEIGHT},
            {"step": _STAGE_FOOTFALL * NUM_STEPS_PER_ENV, "weight": dance_mdp.ENERGY_WEIGHT * 0.5},
            {"step": _STAGE_TEMPO_WIDE * NUM_STEPS_PER_ENV, "weight": dance_mdp.ENERGY_WEIGHT * 0.25},
        ],
    )
    cfg.curriculum["beat_sway_power_weight"] = _stage(
        "beat_sway_power",
        [
            {"step": 0, "weight": 0.0},
            {"step": _STAGE_SWAY * NUM_STEPS_PER_ENV, "weight": dance_mdp.POWER_WEIGHT * 0.25},
            {"step": _STAGE_FOOTFALL * NUM_STEPS_PER_ENV, "weight": dance_mdp.POWER_WEIGHT * 0.5},
        ],
    )
    cfg.curriculum["beat_sway_weight"] = _stage(
        "beat_sway",
        [
            {"step": 0, "weight": 0.0},
            {"step": _STAGE_SWAY * NUM_STEPS_PER_ENV, "weight": 1.0},
            {"step": _STAGE_FOOTFALL * NUM_STEPS_PER_ENV, "weight": 2.0},
        ],
    )
    cfg.curriculum["beat_footfall_weight"] = _stage(
        "beat_footfall",
        [
            {"step": 0, "weight": 0.0},
            {"step": _STAGE_FOOTFALL * NUM_STEPS_PER_ENV, "weight": FOOTFALL_WEIGHT * 0.5},
            {"step": _STAGE_TEMPO_WIDE * NUM_STEPS_PER_ENV, "weight": FOOTFALL_WEIGHT},
        ],
    )
    cfg.curriculum["foot_alternation_weight"] = _stage(
        "foot_alternation_penalty",
        [
            {"step": 0, "weight": 0.0},
            # Only once striking on the beat exists is there anything to
            # alternate; charging repeats earlier just suppresses stepping.
            {"step": _STAGE_ALTERNATION * NUM_STEPS_PER_ENV, "weight": ALTERNATION_WEIGHT * 0.5},
            {"step": _STAGE_TEMPO_WIDE * NUM_STEPS_PER_ENV, "weight": ALTERNATION_WEIGHT},
        ],
    )
    cfg.curriculum["beat_yaw_weight"] = _stage(
        "beat_yaw",
        [
            {"step": 0, "weight": 0.0},
            {"step": _STAGE_FULL * NUM_STEPS_PER_ENV, "weight": 1.0},
        ],
    )
    # Tempo generalisation comes last: a policy that cannot yet hold 120 BPM
    # gains nothing from being shown 60 and 160.
    cfg.curriculum["beat_tempo_range"] = CurriculumTermCfg(
        func=dance_mdp.beat_tempo_range_curriculum,
        params={
            "command_name": "twist",
            "range_stages": [
                {"step": 0, "bpm_range": (118.0, 122.0)},
                {"step": _STAGE_TEMPO_WIDE * NUM_STEPS_PER_ENV, "bpm_range": (100.0, 140.0)},
                {"step": _STAGE_FULL * NUM_STEPS_PER_ENV, "bpm_range": (BPM_MIN, BPM_MAX)},
            ],
        },
    )
    # Mid-episode tempo changes come last of all: re-locking to a new BPM is a
    # harder problem than tracking a steady one.
    cfg.curriculum["beat_tempo_change_prob"] = CurriculumTermCfg(
        func=dance_mdp.beat_tempo_change_prob_curriculum,
        params={
            "command_name": "twist",
            "prob_stages": [
                {"step": 0, "prob": 0.0},
                {"step": _STAGE_FULL * NUM_STEPS_PER_ENV, "prob": 0.25},
            ],
        },
    )
    # Amplitude envelope widens over the same window, and only at the end does
    # it reach down to zero -- the "stand still on the beat" idle case, which
    # uniform sampling would otherwise essentially never produce.
    cfg.curriculum["dance_amplitude_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {
                    "step": 0,
                    "ranges": ((0.0, 0.0), (0.010, 0.014), (0.012, 0.016), (0.0, 0.0), (0.0, 0.0), (0.10, 0.14)),
                },
                {
                    "step": _STAGE_TEMPO_WIDE * NUM_STEPS_PER_ENV,
                    "ranges": ((0.0, 0.0), (0.006, 0.018), (0.008, 0.022), (0.0, 0.0), (0.0, 0.0), (0.05, 0.20)),
                },
                {
                    "step": _STAGE_FULL * NUM_STEPS_PER_ENV,
                    "ranges": ((0.0, 0.0), (0.0, MAX_SWAY), (0.0, MAX_BOB), (0.0, 0.0), (0.0, 0.0), (0.0, MAX_YAW)),
                },
            ],
        },
    )

    if play:
        # A legible demo: one fixed tempo, no mid-episode changes, mid-envelope
        # amplitudes, and every episode starting on a downbeat.
        cfg.commands["twist"].bpm_range = (120.0, 120.0)
        cfg.commands["twist"].tempo_change_prob = 0.0
        cfg.commands["body_pose"].ranges = (
            (0.0, 0.0), (0.014, 0.014), (0.018, 0.018), (0.0, 0.0), (0.0, 0.0), (0.15, 0.15),
        )

    return cfg


# Same PPO hyperparameters as the velocity task; only the run identity changes.
MicroduckBeatDanceRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="beat_dance",
    run_name="beat_dance",
)

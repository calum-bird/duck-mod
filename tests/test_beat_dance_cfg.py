"""Config-level invariants for the beat-dance task.

These need the real mjlab package (they build the actual env cfg against the
actual robot model) and so skip on a machine that only has torch. They are the
cheap half of the pre-flight check described in docs/training.md -- run them
before committing a GPU to a long run.
"""

import pytest

pytest.importorskip("mjlab", reason="config tests need the mjlab training stack")

from microduck_dance.beat_dance_env_cfg import (  # noqa: E402
    ALTERNATION_WEIGHT,
    FOOTFALL_WEIGHT,
    MAX_BOB,
    MAX_SWAY,
    MAX_YAW,
    NUM_STEPS_PER_ENV,
    make_microduck_beat_dance_env_cfg,
)
from microduck_dance.commands import BeatClockCommandCfg  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_beat_dance_env_cfg()


def test_twist_slot_carries_the_beat_clock(cfg):
    assert isinstance(cfg.commands["twist"], BeatClockCommandCfg)
    # Stage 1 is effectively a single tempo; the curriculum widens it later.
    lo, hi = cfg.commands["twist"].bpm_range
    assert hi - lo <= 5.0


def test_observation_contract_is_still_61_dims(cfg):
    """The runtime hot-swaps policies behind a fixed 61-dim vector.

    A task that grows or drops a command slot cannot be deployed alongside the
    walking policy, so this is the single most important invariant in the file.
    """
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert "head_command" in terms
        assert "body_command" in terms
    assert len(cfg.commands["body_pose"].ranges) == 6


def test_walking_reward_terms_are_gone(cfg):
    # These read the twist slot as a velocity, which it no longer is.
    for name in ("track_linear_velocity", "track_angular_velocity", "air_time", "pose"):
        assert name not in cfg.rewards, name
    # body_pose no longer means a pose target -- it means amplitudes.
    assert "body_pose_tracking" not in cfg.rewards


def test_dance_rewards_are_present_with_the_right_signs(cfg):
    # Tracking rewards return [0, 1] -> positive weight.
    assert cfg.rewards["beat_bob"].weight > 0.0
    # Self-negating penalties (<= 0) also take positive weights, so their
    # logged series stays <= 0.
    assert cfg.rewards["station_keeping_penalty"].weight > 0.0
    assert cfg.rewards["heading_drift_penalty"].weight > 0.0
    # joint_deviation_l1 is cost-style (>= 0) and MUST take a negative weight.
    # Getting this backwards is the double-negative bug that makes a policy
    # farm the violation instead of avoiding it.
    assert cfg.rewards["stance_width_penalty"].weight < 0.0


def test_head_is_free_to_oscillate(cfg):
    # An instantaneous head-tracking tax is what killed stepping upstream; only
    # the DC term should be active here.
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    assert cfg.rewards["head_pose_bias"].weight > 0.0


@pytest.mark.parametrize(
    "curriculum_name,reward_name",
    [
        ("beat_sway_weight", "beat_sway"),
        ("beat_footfall_weight", "beat_footfall"),
        ("foot_alternation_weight", "foot_alternation_penalty"),
        ("beat_yaw_weight", "beat_yaw"),
    ],
)
def test_curriculum_stage_zero_matches_the_initial_weight(cfg, curriculum_name, reward_name):
    """A stage-0 weight that disagrees with the term's initial weight silently
    moves the goalposts on the first curriculum tick -- and looks like a
    mysterious step change in wandb rather than a config bug."""
    stages = cfg.curriculum[curriculum_name].params["weight_stages"]
    assert stages[0]["step"] == 0
    assert stages[0]["weight"] == cfg.rewards[reward_name].weight


def test_skills_are_introduced_in_order(cfg):
    """Bob, then sway, then footfall, then alternation: each stage waits for the
    previous skill to consolidate rather than all arriving at once."""

    def onset(curriculum_name):
        stages = cfg.curriculum[curriculum_name].params["weight_stages"]
        return next(s["step"] for s in stages if s["weight"] != 0.0)

    assert cfg.rewards["beat_bob"].weight > 0.0  # active from step 0
    assert onset("beat_sway_weight") < onset("beat_footfall_weight")
    assert onset("beat_footfall_weight") < onset("foot_alternation_weight")


def test_footfall_weight_reaches_its_designed_reward_mass(cfg):
    stages = cfg.curriculum["beat_footfall_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == FOOTFALL_WEIGHT
    alt = cfg.curriculum["foot_alternation_weight"].params["weight_stages"]
    assert alt[-1]["weight"] == ALTERNATION_WEIGHT


def test_amplitude_curriculum_ends_at_the_full_envelope_including_zero(cfg):
    stages = cfg.curriculum["dance_amplitude_range"].params["range_stages"]
    final = stages[-1]["ranges"]
    assert final[1] == (0.0, MAX_SWAY)
    assert final[2] == (0.0, MAX_BOB)
    assert final[5] == (0.0, MAX_YAW)
    # x / roll / pitch stay zero-padded: the slot keeps its width, not its meaning.
    assert final[0] == (0.0, 0.0) and final[3] == (0.0, 0.0) and final[4] == (0.0, 0.0)


def test_play_cfg_is_a_legible_demo():
    play = make_microduck_beat_dance_env_cfg(play=True)
    assert play.commands["twist"].bpm_range == (120.0, 120.0)
    assert play.commands["twist"].randomize_phase is False
    assert play.commands["twist"].tempo_change_prob == 0.0


def test_curriculum_steps_are_expressed_in_environment_steps(cfg):
    # common_step_counter counts env steps, not iterations; a stage written in
    # raw iterations would fire ~24x too early.
    stages = cfg.curriculum["beat_footfall_weight"].params["weight_stages"]
    assert stages[1]["step"] % NUM_STEPS_PER_ENV == 0
    assert stages[1]["step"] >= 1000

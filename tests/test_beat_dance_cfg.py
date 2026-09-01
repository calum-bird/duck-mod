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
    for name in ("track_linear_velocity", "track_angular_velocity", "air_time"):
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
        ("beat_bob_weight", "beat_bob"),
        ("beat_bob_energy_weight", "beat_bob_energy"),
        ("beat_sway_power_weight", "beat_sway_power"),
        ("fallen_tax_weight", "fallen_tax"),
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
    """Bob, then FOOTFALL, then alternation, then sway. Stepping comes right
    after the bob exists because it is the stability mechanism, not garnish: a
    biped marching on the beat catches itself twice a second. Runs 2-3 proved
    the reverse order trains a both-feet-planted bobber that falls every ~2.4 s
    (and a fall tax on top of that teaches freezing). The fallen tax must come
    last of all, only once a step pattern exists to stabilise around."""

    def onset(curriculum_name):
        stages = cfg.curriculum[curriculum_name].params["weight_stages"]
        return next(s["step"] for s in stages if s["weight"] != 0.0)

    assert cfg.rewards["beat_bob"].weight > 0.0  # active from step 0
    assert onset("beat_footfall_weight") < onset("foot_alternation_weight")
    assert onset("foot_alternation_weight") <= onset("beat_sway_weight")
    assert onset("fallen_tax_weight") > onset("beat_footfall_weight")
    assert cfg.curriculum["fallen_tax_weight"].params["weight_stages"][-1]["weight"] <= 2.0


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


def test_leg_posture_regulariser_is_kept_but_weakened(cfg):
    """Run 1 deleted `pose` outright and the duck spent ~47% of each episode
    below the gate height: it was the only thing holding the leg pitch chain in
    a sane stance. It is kept now, weak enough not to fight a few degrees of
    knee and ankle, and scoped off the neck/head so the head stays free."""
    assert "pose" in cfg.rewards
    joints = cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any("neck" in pattern and "?!" in pattern or "neck" not in pattern
               for pattern in joints)
    for std_key in ("std_standing", "std_walking", "std_running"):
        if std_key in cfg.rewards["pose"].params:
            for joint in cfg.rewards["pose"].params[std_key]:
                assert "neck" not in joint and "head" not in joint, joint


def test_bob_std_makes_standing_still_a_bad_deal(cfg):
    """The reward must not be substantially satisfiable by holding the mean.

    Run 1 shipped std=0.010 against a ~14 mm amplitude, where standing still
    scored 0.471 against 0.914 for tracking -- a 1.94x edge that did not pay for
    the fall risk, so the policy stood still. Guard the arithmetic, not the
    number: this fails if a future edit loosens std back toward the amplitude.
    """
    from microduck_dance.beat_clock import bob_payoff
    from microduck_dance.beat_dance_env_cfg import MAX_BOB

    std = cfg.rewards["beat_bob"].params["std"]
    typical_amp = MAX_BOB * 0.6
    still, tracking = bob_payoff(typical_amp, std)
    assert tracking / still >= 2.5, (
        f"std={std} lets standing still score {still:.3f} against {tracking:.3f} "
        f"for tracking ({tracking / still:.2f}x) — the policy will stand still"
    )

def test_motion_terms_carry_the_early_incentive(cfg):
    """The probe-validated ordering: terms that pay stillness exactly zero run
    at full weight from step 0, while the position Gaussians — partly
    satisfiable by standing at the reference's mean — start small and ramp in
    once motion exists. Reversing this recreates run 1's trap."""
    from microduck_dance import mdp as dmdp

    assert cfg.rewards["beat_bob_power"].weight == dmdp.POWER_WEIGHT
    assert cfg.rewards["beat_bob_energy"].weight == dmdp.ENERGY_WEIGHT
    assert cfg.rewards["beat_bob"].weight == 1.0  # precision starts SMALL
    stages = cfg.curriculum["beat_bob_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == 4.0            # and ends primary
    decay = cfg.curriculum["beat_bob_energy_weight"].params["weight_stages"]
    assert decay[-1]["weight"] > 0.0              # bootstrap keeps a floor
    assert decay[-1]["weight"] < dmdp.ENERGY_WEIGHT


def test_sway_std_is_tight_relative_to_its_amplitude(cfg):
    # The same loophole beat_bob had: at std == amplitude a statue collects
    # 0.645 of the sway term for free.
    assert cfg.rewards["beat_sway"].params["std"] <= 0.008


# ---------------------------------------------------------------------------
# Head-swing fine-tune (resumes run 2's checkpoint)
# ---------------------------------------------------------------------------

from microduck_dance.beat_dance_env_cfg import (  # noqa: E402
    MicroduckBeatDanceHeadRlCfg,
    make_microduck_beat_dance_head_finetune_cfg,
)

RUN2_FINAL = {
    "beat_bob": 4.0,
    "beat_sway": 2.0,
    "beat_yaw": 0.0,
    "beat_footfall": 40.0,
    "foot_alternation_penalty": 20.0,
    "beat_bob_energy": 0.75,
    "beat_sway_power": 2.0,
    "beat_bob_power": 4.0,
    "fallen_tax": 0.0,
}


@pytest.fixture(scope="module")
def head_cfg():
    return make_microduck_beat_dance_head_finetune_cfg()


def test_head_finetune_freezes_run2_end_state(head_cfg):
    """Resume may or may not restore the curriculum step counter; the config is
    only counter-proof if run 2's schedules are baked in as static weights and
    the schedules themselves are gone."""
    for name, weight in RUN2_FINAL.items():
        assert head_cfg.rewards[name].weight == weight, name
    for name in (
        "beat_bob_weight", "beat_sway_weight", "beat_yaw_weight",
        "beat_footfall_weight", "foot_alternation_weight",
        "beat_bob_energy_weight", "beat_sway_power_weight",
        "fallen_tax_weight", "beat_tempo_range", "beat_tempo_change_prob",
        "dance_amplitude_range",
    ):
        assert name not in head_cfg.curriculum, name


def test_head_ramp_is_the_only_curriculum_and_starts_at_zero(head_cfg):
    stages = head_cfg.curriculum["head_yaw_power_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0
    assert stages[0]["weight"] == head_cfg.rewards["head_yaw_power"].weight == 0.0
    from microduck_dance import mdp as dmdp

    assert stages[-1]["weight"] == dmdp.HEAD_POWER_WEIGHT
    for s in stages:
        assert s["step"] % NUM_STEPS_PER_ENV == 0


def test_head_finetune_keeps_the_61_dim_contract(head_cfg):
    """Same network in, same network out: the checkpoint can only load if no
    observation or command slot changed width."""
    for group in ("actor", "critic"):
        terms = head_cfg.observations[group].terms
        assert "head_command" in terms
        assert "body_command" in terms
    assert len(head_cfg.commands["body_pose"].ranges) == 6


def test_head_finetune_practises_inside_run2s_envelope(head_cfg):
    lo, hi = head_cfg.commands["twist"].bpm_range
    assert (lo, hi) == (100.0, 140.0)  # brackets the 127 BPM demo
    ranges = head_cfg.commands["body_pose"].ranges
    assert ranges[2][0] > 0.0  # bob never commanded to zero: keep stepping alive
    assert ranges[2][1] <= 0.022  # ... and never past run 2's proven fall line


def test_head_rl_cfg_keeps_run2s_discount():
    """The resumed critic learned returns under gamma=0.99; retargeting it to
    0.995 floods early updates with value error masquerading as advantage."""
    assert MicroduckBeatDanceHeadRlCfg.algorithm.gamma == 0.99
    assert MicroduckBeatDanceHeadRlCfg.experiment_name == "beat_dance_head"
    assert MicroduckBeatDanceHeadRlCfg.max_iterations >= 800


def test_head_finetune_play_cfg_pins_the_demo_tempo():
    play = make_microduck_beat_dance_head_finetune_cfg(play=True)
    assert play.commands["twist"].bpm_range == (127.0, 127.0)
    assert play.commands["twist"].tempo_change_prob == 0.0

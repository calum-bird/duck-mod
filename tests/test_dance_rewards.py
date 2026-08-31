"""Reward-term tests: the gates, the rate limits and the EMA cancellation.

These exercise the properties the reward design actually depends on, not just
that the functions run. Each test names the failure mode it is guarding.
"""

import math

import torch

from microduck_dance import beat_clock as bc
from microduck_dance import mdp as dance_mdp
from tests.conftest import StubEnv, amplitudes, twist_at


# --------------------------------------------------------------------------- #
# The hard gate                                                               #
# --------------------------------------------------------------------------- #


def test_gate_is_open_when_standing_tall_and_upright(env):
    assert torch.all(dance_mdp.dance_gate(env) == 1.0)


def test_gate_closes_when_sitting_low():
    # Guards the butt-hop exploit: bouncing on the trunk to the beat must be
    # worth exactly zero, not merely less.
    e = StubEnv(height=0.055)
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes())
    assert torch.all(dance_mdp.dance_gate(e) == 0.0)


def test_gate_closes_when_tipped_over():
    e = StubEnv()
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes())
    e.set_tilt(45.0)
    assert torch.all(dance_mdp.dance_gate(e) == 0.0)


# --------------------------------------------------------------------------- #
# Choreography tracking                                                       #
# --------------------------------------------------------------------------- #


def test_bob_reward_peaks_when_the_trunk_is_on_the_reference():
    e = StubEnv()
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes(bob=0.02))
    # On the downbeat the reference is nominal - amplitude.
    e.set_height(0.095 - 0.02)
    assert torch.allclose(dance_mdp.beat_bob_tracking(e), torch.ones(e.num_envs), atol=1e-5)


def test_bob_reward_falls_off_away_from_the_reference():
    e = StubEnv()
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes(bob=0.02))
    e.set_height(0.095)  # standing at nominal, i.e. not bobbing at all
    assert torch.all(dance_mdp.beat_bob_tracking(e) < 0.05)


def test_bob_reward_is_zero_while_fallen():
    e = StubEnv(height=0.050)
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes(bob=0.02))
    e.set_height(0.095 - 0.02)  # perfect height...
    e.set_tilt(60.0)            # ...but on its face
    assert torch.all(dance_mdp.beat_bob_tracking(e) == 0.0)


def test_bob_amplitude_is_read_from_the_body_pose_slot():
    e = StubEnv()
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes(bob=0.0))
    e.set_height(0.095)
    # Zero commanded amplitude means "stand still on the beat" -- the idle case.
    assert torch.allclose(dance_mdp.beat_bob_tracking(e), torch.ones(e.num_envs), atol=1e-5)


def test_sway_is_measured_along_the_robots_own_lateral_axis():
    # A duck yawed 90 degrees swaying "left" moves along world -x. If the term
    # measured world y it would score this as zero sway.
    e = StubEnv()
    e.set_commands(twist=twist_at(0.25), body_pose=amplitudes(sway=0.015))
    e.set_yaw(math.pi / 2)
    e.set_xy(torch.tensor([[-0.015, 0.0]]).repeat(e.num_envs, 1))
    assert torch.allclose(dance_mdp.beat_sway_tracking(e), torch.ones(e.num_envs), atol=1e-4)


def test_sway_reference_reverses_across_the_bar():
    e = StubEnv()
    e.set_xy(torch.tensor([[0.0, 0.015]]).repeat(e.num_envs, 1))
    # +y offset is on-reference at phase 0.25 and maximally wrong at 0.75.
    e.set_commands(twist=twist_at(0.25), body_pose=amplitudes(sway=0.015))
    good = dance_mdp.beat_sway_tracking(e)
    e.set_commands(twist=twist_at(0.75), body_pose=amplitudes(sway=0.015))
    bad = dance_mdp.beat_sway_tracking(e)
    assert torch.all(good > 0.99)
    assert torch.all(bad < 0.05)


# --------------------------------------------------------------------------- #
# Footfall timing                                                             #
# --------------------------------------------------------------------------- #


def _step(e, phase, contacts):
    e.set_commands(twist=twist_at(phase, num_envs=e.num_envs), body_pose=amplitudes(num_envs=e.num_envs))
    e.set_contacts(contacts)
    return dance_mdp.beat_footfall_reward(e)


def _feet(left, right, n=4):
    c = torch.zeros(n, 2, dtype=torch.bool)
    c[:, 0] = left
    c[:, 1] = right
    return c


def test_footfall_pays_a_strike_on_the_beat():
    e = StubEnv()
    _step(e, 0.90, _feet(False, False))       # airborne, approaching the beat
    r = _step(e, 0.995, _feet(True, False))   # lands essentially on the beat
    assert torch.all(r > 0.9)


def test_footfall_pays_nothing_for_a_strike_between_beats():
    e = StubEnv()
    _step(e, 0.20, _feet(False, False))
    r = _step(e, 0.25, _feet(True, False))    # exactly off-beat
    assert torch.all(r < 1e-3)


def test_footfall_pays_only_on_the_rising_edge():
    # Guards the jackpot pattern: planting a foot on the beat and holding it
    # must not collect for the rest of the beat.
    e = StubEnv()
    _step(e, 0.95, _feet(False, False))
    first = _step(e, 0.99, _feet(True, False))
    held = _step(e, 0.0, _feet(True, False))  # same contact, still on the beat
    assert torch.all(first > 0.9)
    assert torch.all(held == 0.0)


def test_footfall_pays_at_most_once_per_beat():
    # Guards foot-drumming: two strikes inside one beat collect one payout.
    e = StubEnv()
    _step(e, 0.95, _feet(False, False))
    first = _step(e, 0.99, _feet(True, False))
    _step(e, 0.01, _feet(False, False))
    second = _step(e, 0.02, _feet(False, True))  # a second landing, same beat
    assert torch.all(first > 0.9)
    assert torch.all(second == 0.0)


def test_footfall_re_arms_on_the_next_beat():
    e = StubEnv()
    _step(e, 0.95, _feet(False, False))
    _step(e, 0.99, _feet(True, False))
    _step(e, 0.30, _feet(False, False))       # mid-bar: the beat has wrapped
    again = _step(e, 0.49, _feet(False, True))  # strike on the second downbeat
    assert torch.all(again > 0.9)


def test_footfall_pays_nothing_while_fallen():
    e = StubEnv()
    _step(e, 0.95, _feet(False, False))
    e.set_tilt(60.0)
    r = _step(e, 0.99, _feet(True, False))
    assert torch.all(r == 0.0)


def test_alternation_charges_a_repeated_foot_and_not_an_alternating_one():
    e = StubEnv()
    dance_mdp.foot_alternation_penalty(e)                      # prime: no contact
    e.set_contacts(_feet(True, False))
    first = dance_mdp.foot_alternation_penalty(e)              # left lands
    e.set_contacts(_feet(False, False))
    dance_mdp.foot_alternation_penalty(e)                      # lift
    e.set_contacts(_feet(False, True))
    alternating = dance_mdp.foot_alternation_penalty(e)        # right lands
    e.set_contacts(_feet(False, False))
    dance_mdp.foot_alternation_penalty(e)
    e.set_contacts(_feet(False, True))
    repeated = dance_mdp.foot_alternation_penalty(e)           # right again
    assert torch.all(first == 0.0)        # nothing to repeat yet
    assert torch.all(alternating == 0.0)
    assert torch.all(repeated == -1.0)


def test_alternation_treats_both_feet_as_its_own_id():
    # Otherwise "land both feet every beat" is a loophole around alternating.
    e = StubEnv()
    dance_mdp.foot_alternation_penalty(e)
    e.set_contacts(_feet(True, True))
    dance_mdp.foot_alternation_penalty(e)
    e.set_contacts(_feet(False, False))
    dance_mdp.foot_alternation_penalty(e)
    e.set_contacts(_feet(True, True))
    repeated = dance_mdp.foot_alternation_penalty(e)
    assert torch.all(repeated == -1.0)


def test_penalties_are_self_negating():
    # The sign convention: these must never return a positive value, because
    # they are configured with positive weights.
    e = StubEnv()
    e.set_commands(twist=twist_at(0.0), body_pose=amplitudes())
    e.set_xy(torch.tensor([[0.2, -0.1]]).repeat(e.num_envs, 1))
    e.set_yaw(0.8)
    for _ in range(50):
        assert torch.all(dance_mdp.station_keeping_penalty(e) <= 0.0)
        assert torch.all(dance_mdp.heading_drift_penalty(e) <= 0.0)
        assert torch.all(dance_mdp.foot_alternation_penalty(e) <= 0.0)


# --------------------------------------------------------------------------- #
# Station keeping: the EMA must cancel the choreography, not the drift        #
# --------------------------------------------------------------------------- #


def test_station_keeping_ignores_the_sway_it_is_supposed_to_allow():
    # The central claim of the EMA design: swaying +-15 mm at the bar period is
    # not "wandering off", so it must cost essentially nothing. An
    # instantaneous |xy| penalty would charge ~15 mm every step and cancel the
    # sway reward outright.
    e = StubEnv(num_envs=1)
    bar_period = 2.0  # slowest tempo, 60 BPM -- the hardest case for the EMA
    for i in range(2000):
        t = i * e.step_dt
        y = 0.015 * math.sin(2 * math.pi * t / bar_period)
        e.set_xy(torch.tensor([[0.0, y]]))
        penalty = dance_mdp.station_keeping_penalty(e)
    assert abs(penalty.item()) < 0.002, penalty.item()


def test_station_keeping_charges_a_steady_drift():
    e = StubEnv(num_envs=1)
    for _ in range(2000):
        e.set_xy(torch.tensor([[0.10, 0.0]]))
        penalty = dance_mdp.station_keeping_penalty(e)
    assert penalty.item() < -0.09, penalty.item()


def test_heading_drift_ignores_the_commanded_twist_but_charges_rotation():
    e = StubEnv(num_envs=1)
    for i in range(2000):
        t = i * e.step_dt
        e.set_yaw(0.25 * math.sin(2 * math.pi * t / 2.0))
        oscillating = dance_mdp.heading_drift_penalty(e)
    assert abs(oscillating.item()) < 0.03, oscillating.item()

    e2 = StubEnv(num_envs=1)
    for _ in range(2000):
        e2.set_yaw(0.6)
        drifting = dance_mdp.heading_drift_penalty(e2)
    assert drifting.item() < -0.55, drifting.item()


def test_buffers_reset_between_episodes():
    # A stale 'already paid this beat' flag would silently swallow the first
    # strike of every episode.
    e = StubEnv()
    _step(e, 0.95, _feet(False, False))
    _step(e, 0.99, _feet(True, False))
    e.episode_length_buf = torch.ones(e.num_envs, dtype=torch.long)  # fresh reset
    r = _step(e, 0.995, _feet(True, False))
    assert torch.all(r > 0.9)


# --------------------------------------------------------------------------- #
# Curriculum                                                                  #
# --------------------------------------------------------------------------- #


class _Term:
    def __init__(self, cfg):
        self.cfg = cfg


class _Cfg:
    bpm_range = (118.0, 122.0)
    tempo_change_prob = 0.0


def _curriculum_env(step):
    e = StubEnv(common_step_counter=step)
    e.command_manager._terms["twist"] = _Term(_Cfg())
    return e


def test_tempo_range_widens_at_its_stage_and_not_before():
    stages = [
        {"step": 0, "bpm_range": (118.0, 122.0)},
        {"step": 1000, "bpm_range": (100.0, 140.0)},
        {"step": 2000, "bpm_range": (60.0, 160.0)},
    ]
    for step, expected in ((0, (118.0, 122.0)), (999, (118.0, 122.0)), (1500, (100.0, 140.0)), (5000, (60.0, 160.0))):
        e = _curriculum_env(step)
        dance_mdp.beat_tempo_range_curriculum(e, None, "twist", stages)
        assert e.command_manager.get_term("twist").cfg.bpm_range == expected, step


def test_tempo_change_probability_stays_off_until_its_stage():
    stages = [{"step": 0, "prob": 0.0}, {"step": 1000, "prob": 0.25}]
    for step, expected in ((0, 0.0), (999, 0.0), (1000, 0.25)):
        e = _curriculum_env(step)
        dance_mdp.beat_tempo_change_prob_curriculum(e, None, "twist", stages)
        assert e.command_manager.get_term("twist").cfg.tempo_change_prob == expected, step

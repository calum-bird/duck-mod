"""Kernel-level tests for the beat clock: pure math, no simulator."""

import math

import torch

from microduck_dance import beat_clock as bc
from conftest import twist_at


def test_tempo_norm_round_trips_and_clamps():
    bpm = torch.tensor([60.0, 110.0, 160.0])
    assert torch.allclose(bc.norm_to_tempo(bc.tempo_to_norm(bpm)), bpm, atol=1e-4)
    # Endpoints map to the [-1, 1] the observation normaliser expects.
    assert torch.allclose(bc.tempo_to_norm(torch.tensor([60.0, 160.0])), torch.tensor([-1.0, 1.0]))
    # Out-of-envelope tempo saturates rather than blowing past the range.
    assert bc.tempo_to_norm(torch.tensor([500.0])).item() == 1.0


def test_bar_phase_recovers_from_the_cos_sin_pair():
    for phase in (0.0, 0.12, 0.5, 0.87, 0.999):
        got = bc.bar_phase_from_command(twist_at(phase, num_envs=1))
        assert math.isclose(got.item(), phase, abs_tol=1e-5)


def test_beat_phase_runs_at_twice_the_bar_phase():
    bar = torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert torch.allclose(bc.beat_phase(bar), torch.tensor([0.0, 0.5, 0.0, 0.5]))


def test_phase_distance_is_circular():
    # 0.98 is 0.04 from 0.02, not 0.96 -- landing early is as on-beat as late.
    d = bc.phase_distance(torch.tensor([0.98]), torch.tensor([0.02]))
    assert math.isclose(d.item(), 0.04, abs_tol=1e-6)
    assert bc.phase_distance(torch.tensor([0.0]), torch.tensor([0.5])).item() == 0.5


def test_downbeat_proximity_peaks_on_both_downbeats():
    on_first = bc.downbeat_proximity(torch.tensor([0.0]))
    on_second = bc.downbeat_proximity(torch.tensor([0.5]))
    off = bc.downbeat_proximity(torch.tensor([0.25]))
    assert math.isclose(on_first.item(), 1.0, abs_tol=1e-6)
    assert math.isclose(on_second.item(), 1.0, abs_tol=1e-6)
    # Halfway between beats must be worth essentially nothing, or the timing
    # signal is not a timing signal.
    assert off.item() < 1e-3


def test_downbeat_proximity_tolerance_matches_sigma():
    # At exactly one sigma of beat phase the kernel is exp(-1).
    val = bc.downbeat_proximity(torch.tensor([bc.DOWNBEAT_SIGMA / 2.0]))
    assert math.isclose(val.item(), math.exp(-1.0), rel_tol=1e-5)


def test_bob_is_lowest_on_the_beat_and_highest_between():
    amp = torch.tensor([0.02])
    low = bc.bob_reference(torch.tensor([0.0]), amp, 0.095)
    high = bc.bob_reference(torch.tensor([0.25]), amp, 0.095)
    second_beat = bc.bob_reference(torch.tensor([0.5]), amp, 0.095)
    assert math.isclose(low.item(), 0.075, abs_tol=1e-6)
    assert math.isclose(high.item(), 0.115, abs_tol=1e-6)
    # Two bobs per bar: the second downbeat dips just like the first.
    assert math.isclose(second_beat.item(), 0.075, abs_tol=1e-6)


def test_sway_completes_one_cycle_per_bar_and_crosses_centre_on_beats():
    amp = torch.tensor([0.015])
    assert math.isclose(bc.sway_reference(torch.tensor([0.0]), amp).item(), 0.0, abs_tol=1e-6)
    assert math.isclose(bc.sway_reference(torch.tensor([0.25]), amp).item(), 0.015, abs_tol=1e-6)
    assert math.isclose(bc.sway_reference(torch.tensor([0.5]), amp).item(), 0.0, abs_tol=1e-6)
    assert math.isclose(bc.sway_reference(torch.tensor([0.75]), amp).item(), -0.015, abs_tol=1e-6)


def test_stance_side_splits_the_bar_into_two_beats():
    sides = bc.stance_side(torch.tensor([0.0, 0.49, 0.5, 0.99]))
    assert sides.tolist() == [0, 0, 1, 1]


def test_bob_payoff_exposes_the_stand_still_loophole():
    """A tracking Gaussian is partly satisfiable by holding the reference's
    mean. This is the arithmetic that made run 1 stand rigidly still, and the
    guard against loosening std back toward the amplitude."""
    shipped_still, shipped_track = bc.bob_payoff(0.014, 0.010)
    fixed_still, fixed_track = bc.bob_payoff(0.014, 0.006)
    # What run 1 actually shipped: doing nothing scored ~47% of maximum.
    assert 0.45 < shipped_still < 0.50
    assert shipped_track / shipped_still < 2.0
    # Tightening std makes stillness a clearly worse deal.
    assert fixed_still < 0.30
    assert fixed_track / fixed_still > 2.9


def test_bob_payoff_stillness_falls_as_std_tightens():
    stills = [bc.bob_payoff(0.014, s)[0] for s in (0.010, 0.008, 0.006, 0.004)]
    assert stills == sorted(stills, reverse=True)

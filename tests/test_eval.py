"""Eval-metric tests: synthesise trajectories with known properties and check
the metrics recover them. If these are wrong, every run comparison is wrong."""

import math

import torch

from microduck_dance import eval as dance_eval
from microduck_dance.eval import Trajectory


def make_traj(
    bpm: float = 120.0,
    dt: float = 0.02,
    seconds: float = 8.0,
    strike_offset_ms: float = 0.0,
    alternate: bool = True,
    height: float = 0.095,
    bob: float = 0.0,
) -> Trajectory:
    """A trajectory with one foot strike per beat, offset by a known amount."""
    n = int(seconds / dt)
    t = torch.arange(n, dtype=torch.float32) * dt
    bar_period = 120.0 / bpm
    bar_phase = torch.remainder(t / bar_period, 1.0)

    contacts = torch.zeros(n, 2, dtype=torch.bool)
    beat_seconds = 60.0 / bpm
    offset_s = strike_offset_ms / 1000.0
    foot = 0
    k = 0
    while True:
        strike_t = k * beat_seconds + offset_s
        if strike_t >= seconds - dt:
            break
        idx = int(round(strike_t / dt))
        if 0 <= idx < n:
            # A strike is a rising edge: on for a few steps, then off.
            contacts[idx : idx + 3, foot if alternate else 0] = True
            if alternate:
                foot = 1 - foot
        k += 1

    z = torch.full((n,), height)
    if bob:
        z = z - bob * torch.cos(2 * math.pi * torch.remainder(2 * bar_phase, 1.0))
    return Trajectory(
        bpm=bpm,
        dt=dt,
        bar_phase=bar_phase,
        contacts=contacts,
        root_z=z,
        root_xy=torch.zeros(n, 2),
        root_yaw=torch.zeros(n),
        commanded_bob=bob,
    )


def test_signed_offset_reads_early_as_negative():
    # beat phase 0.98 is 0.02 of a beat EARLY for the next downbeat.
    off = dance_eval.signed_beat_offset(torch.tensor([0.98, 0.02, 0.0]))
    assert math.isclose(off[0].item(), -0.02, abs_tol=1e-6)
    assert math.isclose(off[1].item(), 0.02, abs_tol=1e-6)
    assert math.isclose(off[2].item(), 0.0, abs_tol=1e-6)


def test_strike_indices_finds_rising_edges_and_labels_the_foot():
    c = torch.zeros(12, 2, dtype=torch.bool)
    c[2:5, 0] = True            # left lands at 2, held through 4
    c[6:8, 1] = True            # right lands at 6, released at 8
    c[9, :] = True              # BOTH land at 9 (both were off at 8)
    steps, foot = dance_eval.strike_indices(c)
    assert steps.tolist() == [2, 6, 9]
    assert foot.tolist() == [0, 1, 2]


def test_a_foot_already_down_is_not_a_new_strike():
    # Subtle but load-bearing: if one foot is still planted when the other
    # lands, that is a single strike, not a "both". Getting this wrong would
    # inflate the alternation rate on a shuffling policy.
    c = torch.zeros(6, 2, dtype=torch.bool)
    c[1:5, 1] = True            # right planted from 1
    c[3, 0] = True              # left lands at 3 while right is still down
    steps, foot = dance_eval.strike_indices(c)
    assert steps.tolist() == [1, 3]
    assert foot.tolist() == [1, 0]


def test_perfectly_timed_strikes_score_near_zero_error():
    m = dance_eval.evaluate(make_traj(strike_offset_ms=0.0))
    assert m.strikes >= 14
    assert m.timing_mean_abs_ms < 12.0     # bounded by the 20 ms control step
    assert m.on_beat_fraction == 1.0


def test_late_strikes_are_measured_late_with_the_right_sign():
    m = dance_eval.evaluate(make_traj(strike_offset_ms=80.0))
    assert m.timing_median_ms > 60.0       # late reads positive
    assert m.on_beat_fraction < 0.2        # outside the 50 ms tolerance


def test_early_strikes_read_negative():
    m = dance_eval.evaluate(make_traj(strike_offset_ms=-80.0))
    assert m.timing_median_ms < -60.0


def test_timing_error_is_tempo_relative():
    # The same phase offset is fewer milliseconds at a faster tempo.
    slow = dance_eval.evaluate(make_traj(bpm=60.0, strike_offset_ms=100.0))
    fast = dance_eval.evaluate(make_traj(bpm=160.0, strike_offset_ms=100.0))
    # Both are 100 ms late in wall-clock terms, and the metric reports ms, so
    # they should agree — this guards against accidentally reporting phase.
    assert abs(slow.timing_median_ms - fast.timing_median_ms) < 25.0


def test_alternation_rate_separates_a_dance_from_a_foot_tap():
    assert dance_eval.evaluate(make_traj(alternate=True)).alternation_rate == 1.0
    tapping = dance_eval.evaluate(make_traj(alternate=False))
    assert tapping.alternation_rate == 0.0
    assert any("alternation" in n for n in tapping.notes)


def test_bob_amplitude_is_recovered():
    m = dance_eval.evaluate(make_traj(bob=0.020))
    assert 17.0 < m.bob_amplitude_mm < 23.0
    assert abs(m.bob_amplitude_error_mm) < 3.0


def test_drift_reports_max_and_net_separately():
    n = 100
    xy = torch.zeros(n, 2)
    xy[:, 0] = torch.linspace(0.0, 0.20, n)   # walks out 20 cm
    xy[50:, 0] = torch.linspace(0.10, 0.0, n - 50)  # and comes back
    max_d, net_d = dance_eval.drift_cm(xy)
    assert max_d > net_d
    assert net_d < 1.0


def test_yaw_drift_unwraps_past_pi():
    # A full rotation must read ~360 deg, not fold back to ~0.
    yaw = torch.remainder(torch.linspace(0, 2 * math.pi, 200) + math.pi, 2 * math.pi) - math.pi
    assert abs(dance_eval.yaw_drift_deg(yaw) - 360.0) < 15.0


def test_fall_fraction_uses_the_same_floor_as_the_reward_gate():
    z = torch.cat([torch.full((80,), 0.095), torch.full((20,), 0.050)])
    assert math.isclose(dance_eval.fall_fraction(z), 0.20, abs_tol=1e-6)


def test_a_duck_that_never_steps_is_flagged_not_crashed():
    traj = make_traj()
    traj.contacts = torch.zeros_like(traj.contacts)
    m = dance_eval.evaluate(traj)
    assert m.strikes == 0
    assert m.on_beat_fraction == 0.0
    assert any("never stepped" in n for n in m.notes)


def test_report_and_summary_render_a_tempo_sweep():
    results = [dance_eval.evaluate(make_traj(bpm=b)) for b in (60.0, 120.0, 160.0)]
    report = dance_eval.format_report(results)
    assert "BPM" in report and "on-beat" in report
    assert len(report.splitlines()) >= 5
    s = dance_eval.summarize(results)
    assert s["eval/tempi_tested"] == 3
    assert s["eval/on_beat_fraction"] == 1.0

"""Host-side beat source: the deployment clock must match the training clock."""

import math

from microduck_dance.beat_source import BeatCommandSource, TapTempo, tempo_to_norm


def test_command_is_exactly_thirteen_values_in_slot_order():
    src = BeatCommandSource(bpm=120.0, bob=0.018, sway=0.014, yaw=0.15)
    cmd = src.command()
    assert len(cmd) == 13
    # twist(3) + head(4) + body(6); amplitudes ride in body y / z / yaw.
    assert cmd[7] == 0.0            # body x, zero-padded
    assert math.isclose(cmd[8], 0.014)   # sway
    assert math.isclose(cmd[9], 0.018)   # bob
    assert cmd[10] == 0.0 and cmd[11] == 0.0  # roll, pitch zero-padded
    assert math.isclose(cmd[12], 0.15)   # yaw twist


def test_phase_advances_two_beats_per_bar_at_the_right_rate():
    src = BeatCommandSource(bpm=120.0)
    # 120 BPM -> 0.5 s per beat -> 1.0 s per bar.
    assert math.isclose(src.bar_period, 1.0)
    for _ in range(50):
        src.advance(0.01)  # half a bar
    assert math.isclose(src.phase, 0.5, abs_tol=1e-6)


def test_tempo_normalisation_matches_the_training_encoding():
    from microduck_dance import beat_clock
    import torch

    for bpm in (60.0, 95.0, 120.0, 160.0):
        host = tempo_to_norm(bpm)
        trained = beat_clock.tempo_to_norm(torch.tensor([bpm])).item()
        assert math.isclose(host, trained, abs_tol=1e-6), bpm


def test_sync_snaps_the_clock_to_a_downbeat():
    src = BeatCommandSource(bpm=100.0)
    src.advance(0.31)
    src.sync(0.0)
    assert src.phase == 0.0
    assert math.isclose(src.command()[0], 1.0)  # cos(0)
    assert math.isclose(src.command()[1], 0.0, abs_tol=1e-9)


def test_set_tempo_keeps_phase_continuous_and_clamps_to_the_trained_envelope():
    src = BeatCommandSource(bpm=120.0)
    src.advance(0.4)
    before = src.phase
    src.set_tempo(300.0)
    assert src.phase == before          # no phase jump
    assert src.bpm == 160.0             # clamped to what the policy was trained on
    src.set_tempo(10.0)
    assert src.bpm == 60.0


def test_tap_tempo_averages_intervals():
    tap = TapTempo()
    assert tap.tap(0.0) is None         # one tap is not a tempo
    tap.tap(0.5)
    tap.tap(1.0)
    bpm = tap.tap(1.5)
    assert math.isclose(bpm, 120.0, abs_tol=1e-6)


def test_tap_tempo_restarts_after_a_long_gap():
    tap = TapTempo(timeout=2.0)
    tap.tap(0.0)
    tap.tap(0.5)
    assert tap.tap(10.0) is None        # gap > timeout starts a fresh measurement

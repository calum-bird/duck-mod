"""The --song flow: tempo folding into the trained band, playback alignment,
and the controller hook that starts audio on the sim's first step."""

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import song_sync  # noqa: E402


def test_in_band_tempo_is_used_verbatim():
    bpm, note = song_sync.choose_drive_bpm(120.0)
    assert bpm == 120.0 and note is None


@pytest.mark.parametrize(
    "song,drive",
    [(70.0, 140.0), (65.0, 130.0), (33.0, 132.0), (260.0, 130.0)],
)
def test_out_of_band_tempo_folds_by_powers_of_two(song, drive):
    # Only 2**k ratios keep the duck on the song's grid — any other ratio
    # walks the beat across the bar, which looks worse than a wrong tempo.
    bpm, note = song_sync.choose_drive_bpm(song)
    assert bpm == pytest.approx(drive)
    assert note is not None


def test_unfoldable_tempo_is_driven_raw_with_a_warning():
    # The band ratio (1.4) is under 2, so some tempos have no fit at all.
    bpm, note = song_sync.choose_drive_bpm(150.0)
    assert bpm == 150.0
    assert "raw" in note


def test_trim_cuts_exactly_the_pre_downbeat_lead_in(tmp_path):
    import soundfile as sf

    sr = 8000
    path = tmp_path / "t.wav"
    sf.write(str(path), np.zeros(sr * 2, dtype=np.float32), sr)
    out = song_sync.trimmed_for_playback(str(path), 0.5)
    assert sf.info(out).frames == int(1.5 * sr)
    # A sub-perceptual offset is not worth a temp file.
    assert song_sync.trimmed_for_playback(str(path), 0.01) == str(path)


def test_controller_fires_on_start_exactly_once_at_first_tick():
    from microduck_dance.dance_driver import BeatController

    calls = []
    c = BeatController(bpm=120.0, on_start=lambda: calls.append(1))
    assert calls == []  # construction is minutes before the sim loop; no fire
    c.advance(0.02)
    c.advance(0.02)
    assert calls == [1]

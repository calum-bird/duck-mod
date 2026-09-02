"""Everything the dance scripts need to follow a song file.

Measures the track (via beat_track), picks a drive tempo inside the policy's
trained band, and — for the live viewer — prepares an aligned playback copy
and starts a player. Sibling-script import: lives next to beat_track.py and
is imported by play_dance.py / render_dance.py, which run from this directory.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import beat_track

# Where the policy actually practised (and eval showed 0% falls). Outside it
# the dance degrades gracefully rather than falling over, but footwork gets
# sloppier the further out you go.
TRAINED_BAND = (100.0, 140.0)
STEADINESS_FLOOR = 3.0


def analyze(path: str, bpm_min: float = 60.0, bpm_max: float = 170.0):
    """(bpm, steadiness, first_beat_offset_s) for a track."""
    tempo, steadiness, offset = beat_track.estimate(str(path), bpm_min, bpm_max)
    if steadiness < STEADINESS_FLOOR:
        print(
            f"! unsteady pulse (score {steadiness:.1f}) — live drums or rubato "
            "drift off the fixed clock; expect the duck to fall behind",
            file=sys.stderr,
        )
    return tempo, steadiness, offset


def choose_drive_bpm(song_bpm: float, band=TRAINED_BAND) -> tuple[float, str | None]:
    """Pick song_bpm * 2**k inside the trained band, preferring the song's own
    tempo. Only powers of two keep the duck on the song's grid — any other
    ratio walks the beat across the bar."""
    lo, hi = band
    if lo <= song_bpm <= hi:
        return song_bpm, None
    words = {1: "double-time", -1: "half-time", 2: "quadruple-time", -2: "quarter-time"}
    for k in (1, -1, 2, -2):
        cand = song_bpm * (2.0**k)
        if lo <= cand <= hi:
            return cand, f"song is {song_bpm:.1f} BPM — dancing {words[k]} at {cand:.1f}"
    return song_bpm, (
        f"song is {song_bpm:.1f} BPM with no power-of-two fit in the trained "
        f"{lo:.0f}-{hi:.0f} band — driving it raw; expect sloppier footwork"
    )


def trimmed_for_playback(path: str, offset_s: float) -> str:
    """A copy with everything before the first downbeat cut off, so playback
    started at sim t=0 lands beat 1 on the clock's beat 1."""
    if offset_s < 0.02:
        return str(path)
    import soundfile as sf

    y, sr = sf.read(str(path), always_2d=True)
    out = Path(tempfile.mkdtemp(prefix="dance_song_")) / (
        Path(path).stem + "_aligned.wav"
    )
    sf.write(str(out), y[int(offset_s * sr) :], sr)
    return str(out)


def spawn_player(path: str) -> subprocess.Popen | None:
    """Start audio playback: afplay on macOS, ffplay elsewhere; None if neither
    is on PATH (the viewer then runs silent)."""
    for cmd in (
        ["afplay", str(path)],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
    ):
        if shutil.which(cmd[0]):
            return subprocess.Popen(cmd)
    print("no afplay/ffplay on PATH — running silent", file=sys.stderr)
    return None

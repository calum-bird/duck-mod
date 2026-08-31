#!/usr/bin/env python3
"""Estimate a track's tempo, steadiness, and first-downbeat offset. numpy only.

    python scripts/beat_track.py song.wav [--bpm-min 60 --bpm-max 170]

Exists because the dance policy is commanded in BPM and the video muxes audio
so the clock's beat 1 lands on the track's first downbeat -- both numbers have
to come from the audio itself, and a track with an unsteady pulse (live drums,
rubato) should be rejected here rather than discovered as a drifting duck.

Method: half-wave-rectified spectral-flux onset envelope at ~86 Hz, then
autocorrelation over the BPM window for tempo (with a steadiness score from
the peak's prominence), then the onset phase, folded at the beat period, for
the offset of the first strong beat.
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

HOP = 256


def onset_envelope(path: str) -> tuple[np.ndarray, float]:
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1).astype(np.float32)
    win = 512
    n = (len(y) - win) // HOP
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n, win), strides=(y.strides[0] * HOP, y.strides[0])
    )
    mag = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
    flux -= flux.mean()
    return flux, sr / HOP


def estimate(path: str, bpm_min: float, bpm_max: float):
    env, fps = onset_envelope(path)
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(1, len(ac))
    bpm = 60.0 * fps / lags
    band = (bpm >= bpm_min) & (bpm <= bpm_max)
    ac_band = ac[1:][band]
    best = np.argmax(ac_band)
    tempo = float(bpm[band][best])
    # Steadiness: the tempo peak against the band's typical level.
    steadiness = float(ac_band[best] / (np.abs(ac_band).mean() + 1e-9))

    # Phase: fold onset energy at the beat period, take the strongest bin.
    period = 60.0 * fps / tempo
    t = np.arange(len(env))
    bins = 32
    phase_energy = np.zeros(bins)
    idx = ((t % period) / period * bins).astype(int) % bins
    np.add.at(phase_energy, idx, np.maximum(env, 0))
    phase = (np.argmax(phase_energy) + 0.5) / bins
    first_beat_s = float(phase * period / fps)
    return tempo, steadiness, first_beat_s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--bpm-min", type=float, default=60.0)
    p.add_argument("--bpm-max", type=float, default=170.0)
    args = p.parse_args()
    print(f"{'track':<58}{'BPM':>7}{'steady':>8}{'beat1 s':>9}")
    for path in args.paths:
        tempo, steady, off = estimate(path, args.bpm_min, args.bpm_max)
        name = path.split("/")[-1][:56]
        print(f"{name:<58}{tempo:7.1f}{steady:8.1f}{off:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

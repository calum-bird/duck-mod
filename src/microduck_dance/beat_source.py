"""Host-side beat source: turns a tempo into the 13 command values the policy
expects, at whatever rate you call it.

This is the deployment counterpart of the training-time ``BeatClockCommand``.
On the robot the policy still reads the same 61-dim observation vector; someone
just has to fill the command half of it with a clock that matches the music.
That someone is this class, running on the host (or on the duck) next to
whatever produces the tempo -- a tap-tempo button, a beat tracker over the
microphone, a DAW clock, or a hard-coded BPM.

Deliberately dependency-free (no torch, no mjlab): it must be able to run on a
laptop, a Raspberry Pi, or inside the robot's own Python without dragging the
training stack along.

Transport is *not* included. The microduck runtime takes commands over a
JSON-RPC interface on a Unix socket (``robotctl`` on-device, ``duckctl``
remotely); wire :meth:`BeatCommandSource.command` into whichever RPC method
your runtime version exposes for policy commands, rather than trusting a
guessed method name from this file.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

BPM_MIN = 60.0
BPM_MAX = 160.0


def tempo_to_norm(bpm: float, bpm_min: float = BPM_MIN, bpm_max: float = BPM_MAX) -> float:
    """Same normalisation the policy was trained with. Must match, or the
    policy will read every tempo as the wrong tempo."""
    span = max(bpm_max - bpm_min, 1e-6)
    return max(-1.0, min(1.0, 2.0 * (bpm - bpm_min) / span - 1.0))


@dataclass
class BeatCommandSource:
    """Produces ``[twist(3), head_pose(4), body_pose(6)]`` for the dance policy.

    The bar phase advances at ``bpm``; one bar is two beats, matching training.
    Amplitudes ride in the body_pose slot (y -> sway, z -> bob, yaw -> twist)
    and the head slot carries a static attitude the policy holds on average
    while it bobs.
    """

    bpm: float = 120.0
    bob: float = 0.018
    sway: float = 0.014
    yaw: float = 0.15
    head_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    bpm_min: float = BPM_MIN
    bpm_max: float = BPM_MAX
    _phase: float = field(default=0.0, repr=False)

    @property
    def phase(self) -> float:
        """Current bar phase in [0, 1). Two beats per bar."""
        return self._phase

    @property
    def bar_period(self) -> float:
        return 120.0 / max(self.bpm, 1e-6)

    def advance(self, dt: float) -> None:
        self._phase = (self._phase + dt / self.bar_period) % 1.0

    def sync(self, phase: float = 0.0) -> None:
        """Snap the clock to a phase -- call with 0.0 on a detected downbeat.

        Worth doing sparingly: a hard snap every beat makes the phase signal
        jitter, and the policy was trained on a phase that only ever advances
        smoothly.
        """
        self._phase = phase % 1.0

    def set_tempo(self, bpm: float) -> None:
        """Change tempo without breaking phase continuity, exactly as the
        training-time mid-episode tempo change does."""
        self.bpm = max(self.bpm_min, min(self.bpm_max, bpm))

    def command(self) -> list[float]:
        """The 13 command values, in the runtime's fixed slot order."""
        two_pi_phase = 2.0 * math.pi * self._phase
        twist = [
            math.cos(two_pi_phase),
            math.sin(two_pi_phase),
            tempo_to_norm(self.bpm, self.bpm_min, self.bpm_max),
        ]
        body = [0.0, self.sway, self.bob, 0.0, 0.0, self.yaw]
        return twist + list(self.head_pose) + body

    def tick(self, dt: float) -> list[float]:
        self.advance(dt)
        return self.command()


class TapTempo:
    """Estimate BPM from taps (a button, a key, a footswitch).

    Averages the last few intervals rather than using the most recent one, so a
    single sloppy tap does not throw the tempo; taps more than ``timeout``
    apart start a new measurement.
    """

    def __init__(self, window: int = 4, timeout: float = 2.5):
        self.window = window
        self.timeout = timeout
        self._taps: list[float] = []

    def tap(self, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        if self._taps and now - self._taps[-1] > self.timeout:
            self._taps = []
        self._taps.append(now)
        if len(self._taps) < 2:
            return None
        self._taps = self._taps[-(self.window + 1):]
        intervals = [b - a for a, b in zip(self._taps, self._taps[1:])]
        mean = sum(intervals) / len(intervals)
        return 60.0 / mean if mean > 0 else None


def main() -> int:
    """Print the command stream for a given tempo -- a dry run of what the
    runtime would be fed, useful for eyeballing the clock before deploying."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--rate", type=float, default=50.0, help="control rate (Hz)")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--bob", type=float, default=0.018)
    parser.add_argument("--sway", type=float, default=0.014)
    parser.add_argument("--yaw", type=float, default=0.15)
    args = parser.parse_args()

    src = BeatCommandSource(bpm=args.bpm, bob=args.bob, sway=args.sway, yaw=args.yaw)
    dt = 1.0 / args.rate
    for i in range(int(args.seconds * args.rate)):
        cmd = src.tick(dt)
        beat_phase = (2.0 * src.phase) % 1.0
        marker = "  <- beat" if beat_phase < (2.0 * dt / src.bar_period) else ""
        print(f"t={i * dt:5.2f}s  phase={src.phase:.3f}  cmd={[round(c, 3) for c in cmd]}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

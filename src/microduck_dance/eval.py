"""Headless evaluation metrics for the beat-dance policy.

Upstream's playbook is blunt about why this exists: *"Before theorizing
failures, run headless eval batteries... 'rolls but face-plants 1 in 3' is
measurable; 'it works' is not."* Reward curves cannot tell you whether a duck
is dancing. These metrics can.

Everything here is a **pure function over a recorded trajectory** — no
simulator, no policy, no GPU — so the metric logic is unit-tested on CPU and
the only thing that can go wrong on the node is the recording itself.
``scripts/eval_dance.py`` does the recording and calls into here.

The headline number is **beat-timing error in milliseconds**: for every foot
strike, how far from the nearest downbeat it landed, signed so early reads
negative. That single distribution is what separates "dances" from "moves
rhythmically-ish", and it is directly comparable across tempi, across reward
variants, and against the ~40 ms the reward's Gaussian was shaped around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from microduck_dance import beat_clock


# --------------------------------------------------------------------------- #
# Trajectory container                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Trajectory:
    """One recorded rollout at a fixed tempo.

    All tensors are (T,) except ``contacts``, which is (T, 2) boolean
    left/right foot contact in the sensor's own order.
    """

    bpm: float
    dt: float
    bar_phase: torch.Tensor
    contacts: torch.Tensor
    root_z: torch.Tensor
    root_xy: torch.Tensor
    root_yaw: torch.Tensor
    commanded_bob: float = 0.0
    commanded_sway: float = 0.0

    def __len__(self) -> int:
        return int(self.bar_phase.shape[0])


@dataclass
class DanceMetrics:
    """What a run is actually worth, in numbers you can compare between runs."""

    bpm: float
    strikes: int
    timing_mean_abs_ms: float
    timing_median_ms: float
    timing_p90_abs_ms: float
    timing_jitter_ms: float
    on_beat_fraction: float
    alternation_rate: float
    strikes_per_beat: float
    bob_amplitude_mm: float
    bob_amplitude_error_mm: float
    sway_amplitude_mm: float
    max_drift_cm: float
    net_drift_cm: float
    yaw_drift_deg: float
    fall_fraction: float
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Timing                                                                      #
# --------------------------------------------------------------------------- #


def signed_beat_offset(beat_phase: torch.Tensor) -> torch.Tensor:
    """Offset from the nearest downbeat in beats, in [-0.5, 0.5).

    Signed on purpose: a policy that is consistently 30 ms *early* is a
    different problem from one that is 30 ms *late* (the first is anticipating
    the clock, the second is lagging the actuators), and an absolute error
    hides which one you have.
    """
    return torch.remainder(beat_phase + 0.5, 1.0) - 0.5


def strike_indices(contacts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rising-edge foot strikes.

    Returns ``(step_index, foot_id)`` where foot_id is 0 left, 1 right, 2 both
    in the same step. Matches the convention the reward's alternation term uses,
    so eval and training agree on what "a strike" is.
    """
    if contacts.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    prev = torch.zeros_like(contacts)
    prev[1:] = contacts[:-1]
    new = contacts & ~prev
    left, right = new[:, 0], new[:, 1]
    landed = left | right
    steps = torch.nonzero(landed, as_tuple=False).flatten()
    both = left[steps] & right[steps]
    foot = torch.where(both, torch.full_like(steps, 2), right[steps].long())
    return steps, foot


def timing_errors_ms(traj: Trajectory) -> torch.Tensor:
    """Signed milliseconds from the nearest downbeat, one per strike."""
    steps, _ = strike_indices(traj.contacts)
    if steps.numel() == 0:
        return torch.zeros(0)
    b = beat_clock.beat_phase(traj.bar_phase[steps])
    beat_seconds = 60.0 / max(traj.bpm, 1e-6)
    return signed_beat_offset(b) * beat_seconds * 1000.0


def alternation_rate(traj: Trajectory) -> float:
    """Fraction of consecutive strikes that changed foot.

    1.0 is a clean left-right-left. Near 0.5 with a two-beat pattern usually
    means one foot is tapping while the other stays planted — on beat, but not
    a dance, which is exactly what the alternation penalty exists to stop.
    """
    _, foot = strike_indices(traj.contacts)
    if foot.numel() < 2:
        return float("nan")
    changed = (foot[1:] != foot[:-1]).float()
    return float(changed.mean())


# --------------------------------------------------------------------------- #
# Posture and station keeping                                                 #
# --------------------------------------------------------------------------- #


def _peak_to_peak_amplitude(x: torch.Tensor, drop_fraction: float = 0.1) -> float:
    """Robust half peak-to-peak: the 5th-95th percentile spread, halved.

    Percentiles rather than min/max because a single contact spike or one
    stumble would otherwise define the "amplitude" of an entire run.
    """
    if x.numel() == 0:
        return 0.0
    lo = torch.quantile(x, drop_fraction / 2)
    hi = torch.quantile(x, 1.0 - drop_fraction / 2)
    return float((hi - lo) / 2)


def drift_cm(root_xy: torch.Tensor) -> tuple[float, float]:
    """(max excursion, net displacement) from the starting position, in cm."""
    if root_xy.shape[0] == 0:
        return 0.0, 0.0
    rel = root_xy - root_xy[0]
    dist = torch.norm(rel, dim=-1)
    return float(dist.max() * 100.0), float(dist[-1] * 100.0)


def yaw_drift_deg(root_yaw: torch.Tensor) -> float:
    """Net rotation from the start, unwrapped, in degrees.

    Unwrapped so a duck that slowly pirouettes past +/-pi reads as 400 degrees
    rather than folding back to near zero.
    """
    if root_yaw.numel() < 2:
        return 0.0
    d = torch.remainder(root_yaw[1:] - root_yaw[:-1] + torch.pi, 2 * torch.pi) - torch.pi
    return float(torch.rad2deg(d.sum()))


def fall_fraction(root_z: torch.Tensor, min_height: float = 0.070) -> float:
    """Fraction of steps spent below the dance gate's height floor.

    Deliberately the same threshold the reward gate uses, so "the gate was
    closed" in training and "it fell" in eval mean the same thing.
    """
    if root_z.numel() == 0:
        return 0.0
    return float((root_z < min_height).float().mean())


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #


def evaluate(traj: Trajectory, on_beat_tolerance_ms: float = 50.0) -> DanceMetrics:
    errs = timing_errors_ms(traj)
    duration = len(traj) * traj.dt
    beats = duration * traj.bpm / 60.0

    notes: list[str] = []
    if errs.numel() == 0:
        notes.append("no foot strikes recorded — the duck never stepped")
    if errs.numel() and float(errs.abs().mean()) > 120.0:
        notes.append("timing error exceeds a third of a beat: not locked to the clock")
    alt = alternation_rate(traj)
    if alt == alt and alt < 0.5:  # not NaN
        notes.append("low alternation: likely tapping one foot")
    fall = fall_fraction(traj.root_z)
    if fall > 0.05:
        notes.append(f"below the gate height for {fall:.0%} of the run")
    max_drift, net_drift = drift_cm(traj.root_xy)
    if max_drift > 15.0:
        notes.append(f"wandered {max_drift:.0f} cm — station keeping is not holding")

    bob = _peak_to_peak_amplitude(traj.root_z)
    sway = _peak_to_peak_amplitude(traj.root_xy[:, 1]) if traj.root_xy.shape[0] else 0.0

    return DanceMetrics(
        bpm=traj.bpm,
        strikes=int(errs.numel()),
        timing_mean_abs_ms=float(errs.abs().mean()) if errs.numel() else float("nan"),
        timing_median_ms=float(errs.median()) if errs.numel() else float("nan"),
        timing_p90_abs_ms=(
            float(torch.quantile(errs.abs(), 0.9)) if errs.numel() else float("nan")
        ),
        timing_jitter_ms=float(errs.std()) if errs.numel() > 1 else float("nan"),
        on_beat_fraction=(
            float((errs.abs() <= on_beat_tolerance_ms).float().mean())
            if errs.numel()
            else 0.0
        ),
        alternation_rate=alt,
        strikes_per_beat=(float(errs.numel()) / beats) if beats > 0 else 0.0,
        bob_amplitude_mm=bob * 1000.0,
        bob_amplitude_error_mm=(bob - traj.commanded_bob) * 1000.0,
        sway_amplitude_mm=sway * 1000.0,
        max_drift_cm=max_drift,
        net_drift_cm=net_drift,
        yaw_drift_deg=yaw_drift_deg(traj.root_yaw),
        fall_fraction=fall,
        notes=notes,
    )


def format_report(results: list[DanceMetrics]) -> str:
    """A tempo-sweep table. This is the artifact to paste into a run comparison."""
    header = (
        f"{'BPM':>5} {'strikes':>8} {'|err|ms':>8} {'p90ms':>7} {'jitter':>7} "
        f"{'on-beat':>8} {'alt':>6} {'/beat':>6} {'bob mm':>7} {'drift cm':>9} {'fall':>6}"
    )
    lines = [header, "-" * len(header)]
    for m in results:
        lines.append(
            f"{m.bpm:5.0f} {m.strikes:8d} {m.timing_mean_abs_ms:8.1f} "
            f"{m.timing_p90_abs_ms:7.1f} {m.timing_jitter_ms:7.1f} "
            f"{m.on_beat_fraction:7.0%} {m.alternation_rate:6.2f} "
            f"{m.strikes_per_beat:6.2f} {m.bob_amplitude_mm:7.1f} "
            f"{m.max_drift_cm:9.1f} {m.fall_fraction:5.0%}"
        )
    notes = {n for m in results for n in m.notes}
    if notes:
        lines.append("")
        lines.extend(f"  ! {n}" for n in sorted(notes))
    return "\n".join(lines)


def summarize(results: list[DanceMetrics]) -> dict:
    """Flat dict for wandb / mission-chart logging."""
    ok = [m for m in results if m.strikes > 0]
    return {
        "eval/tempi_tested": len(results),
        "eval/tempi_with_strikes": len(ok),
        "eval/timing_mean_abs_ms": (
            sum(m.timing_mean_abs_ms for m in ok) / len(ok) if ok else float("nan")
        ),
        "eval/on_beat_fraction": (
            sum(m.on_beat_fraction for m in ok) / len(ok) if ok else 0.0
        ),
        "eval/alternation_rate": (
            sum(m.alternation_rate for m in ok) / len(ok) if ok else float("nan")
        ),
        "eval/max_drift_cm": max((m.max_drift_cm for m in results), default=0.0),
        "eval/fall_fraction": max((m.fall_fraction for m in results), default=0.0),
    }

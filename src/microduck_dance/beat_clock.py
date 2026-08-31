"""Pure phase/tempo kernels for the beat-synced dance task.

This module is deliberately free of ``mjlab`` imports — it needs only
``torch`` — so the choreography math can be unit-tested on a CPU box with no
simulator, no GPU and no robot. The mjlab-facing pieces live in
``commands.py`` (the command term) and ``mdp.py`` (the reward terms).

Timebase
--------
The clock's phase is a **bar** phase, where one bar = two beats:

    bar_phase φ ∈ [0, 1)      one bar   = 2 beats = 120 / bpm seconds
    beat_phase b = (2φ) mod 1 one beat  =           60 / bpm seconds

Downbeats (the moments a foot should strike) are at φ = 0 and φ = 0.5.

Two beats rather than one is the natural unit here because the duck is a
biped: the weight shifts left on one beat and right on the next, so the
lateral sway has a *bar* period while the vertical bob has a *beat* period.
A single-beat clock cannot express "which foot" without extra state.

Encoding into the twist slot
----------------------------
The runtime observation vector is frozen at 61 dims — 48 proprioception plus
13 command values laid out ``[twist(3), head_pose(4), body_pose(6)]`` — and
policies hot-swap behind that contract, so a new command cannot simply be
appended. This task therefore **repurposes the 3-wide twist slot** as the beat
clock, exactly as the shipped ground-pick/sit-stand/spin tasks repurpose it as
a phase signal (see ``GroundPickPhaseCommand`` upstream):

    twist[0] = cos(2πφ)          bar phase, as a continuous pair so the
    twist[1] = sin(2πφ)          policy sees no discontinuity at the wrap
    twist[2] = tempo_norm        BPM mapped to [-1, 1]

The phase is sent as (cos, sin) rather than as a raw ramp because a sawtooth
has a step discontinuity at every bar boundary, which a policy has to spend
capacity learning to ignore.
"""

from __future__ import annotations

import math

import torch

# Tempo envelope the policy is trained over. 60-160 BPM covers most dance
# music; the bar period is 120/bpm s, so 2.0 s at 60 BPM down to 0.75 s at 160.
BPM_MIN = 60.0
BPM_MAX = 160.0

# Default timing tolerance for "on the beat", in beat-phase units. At 120 BPM a
# beat is 500 ms, so sigma=0.08 is a 40 ms tolerance -- the order of human
# beat-timing accuracy, and tempo-relative by construction.
DOWNBEAT_SIGMA = 0.08

TWO_PI = 2.0 * math.pi


def tempo_to_norm(bpm: torch.Tensor, bpm_min: float = BPM_MIN, bpm_max: float = BPM_MAX) -> torch.Tensor:
    """Map BPM to the [-1, 1] range carried in twist[2].

    Normalised rather than raw so the command slot stays in the same numeric
    range as every other observation the policy sees (raw BPM would be a
    ~100x-larger input than anything else in the vector and would dominate the
    first layer until the normaliser caught up).
    """
    span = max(bpm_max - bpm_min, 1e-6)
    return (2.0 * (bpm - bpm_min) / span - 1.0).clamp(-1.0, 1.0)


def norm_to_tempo(norm: torch.Tensor, bpm_min: float = BPM_MIN, bpm_max: float = BPM_MAX) -> torch.Tensor:
    """Inverse of :func:`tempo_to_norm` (used by the reward terms and the host)."""
    span = bpm_max - bpm_min
    return bpm_min + (norm.clamp(-1.0, 1.0) + 1.0) * 0.5 * span


def bar_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """Recover bar phase φ ∈ [0, 1) from the (cos, sin) pair in twist[0:2]."""
    phase = torch.atan2(cmd[:, 1], cmd[:, 0]) / TWO_PI
    return torch.remainder(phase, 1.0)


def beat_phase(bar_phase: torch.Tensor) -> torch.Tensor:
    """Beat phase b ∈ [0, 1): two beats per bar, so b = 2φ (mod 1)."""
    return torch.remainder(2.0 * bar_phase, 1.0)


def phase_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular distance between two phases in [0, 1), result in [0, 0.5].

    Circular because phase 0.98 is 0.04 away from phase 0.02, not 0.96 -- a
    foot landing a hair *early* is as on-beat as one landing a hair late, and a
    linear distance would score it as maximally wrong.
    """
    d = torch.remainder(a - b, 1.0)
    return torch.minimum(d, 1.0 - d)


def downbeat_proximity(bar_phase: torch.Tensor, sigma: float = DOWNBEAT_SIGMA) -> torch.Tensor:
    """Gaussian kernel in [0, 1] peaking at every downbeat (φ = 0 and φ = 0.5).

    Used to score *when* a foot strike happens. Gaussian rather than a hard
    window so there is a gradient pulling a mistimed strike toward the beat
    instead of a cliff that pays nothing until the policy stumbles into the
    window by chance.
    """
    b = beat_phase(bar_phase)
    d = phase_distance(b, torch.zeros_like(b))
    return torch.exp(-((d / max(sigma, 1e-6)) ** 2))


def bob_reference(bar_phase: torch.Tensor, amplitude: torch.Tensor, nominal_z: float) -> torch.Tensor:
    """Target trunk height: dips to its lowest exactly ON each downbeat.

    z_ref = nominal_z - A·cos(2π·beat_phase), so at b = 0 the target is
    nominal_z - A (compressed, weight landing) and at b = 0.5 it is
    nominal_z + A (extended, between beats). One full bob per beat.
    """
    return nominal_z - amplitude * torch.cos(TWO_PI * beat_phase(bar_phase))


def sway_reference(bar_phase: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """Target lateral offset: one full left-right cycle per BAR (two beats).

    +A at φ = 0.25 (the middle of the first beat) and -A at φ = 0.75, so the
    weight is committed to one side per beat and crosses through centre at each
    downbeat -- the shape a biped actually makes when stepping in place.
    """
    return amplitude * torch.sin(TWO_PI * bar_phase)


def yaw_reference(bar_phase: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """Target trunk yaw: turns into the sway, same bar period."""
    return amplitude * torch.sin(TWO_PI * bar_phase)


def stance_side(bar_phase: torch.Tensor) -> torch.Tensor:
    """Which half of the bar we are in: 0 for the first beat, 1 for the second.

    The foot-alternation term uses this to know which strike belongs to which
    beat without needing a monotonic beat counter.
    """
    return (bar_phase >= 0.5).long()

#!/usr/bin/env python3
"""Decide, from a training log, whether a run is worth continuing.

The first few hundred iterations of a run carry almost all of the information
about whether the REWARD DESIGN is sound -- and none of that information is in
the final policy, because a badly-signed reward produces a confident, stable,
completely wrong duck. This turns "does it look okay?" into a verdict with
named conditions, so a run is killed for a reason rather than a feeling.

    python scripts/pilot_check.py train.log --min-iters 300

Conditions, in the order they matter:

FAIL  a reward term whose name says penalty is logging POSITIVE. This is the
      double-negative bug from upstream's playbook: a positive weight on a
      cost-style return pays the policy to commit the violation. Nothing else
      in the run is interpretable until it is fixed.
FAIL  any NaN termination. Physics has diverged; the checkpoint is worthless.
WARN  the stage-1 objective (beat_bob) is not rising. It is the only positive
      task term active early, so if it is flat nothing downstream can work.
WARN  falls are not decreasing. The duck must learn to stand before it can
      learn to dance on the beat.
WARN  policy entropy collapsing. Upstream watched a run go degenerate this way
      (10.9 -> 1.9) while its reward curve still looked fine.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

# Trainer prints "        Some/Metric_name: 0.1234" inside a block per iteration.
METRIC = re.compile(r"^\s*([A-Za-z][\w/ .-]*?):\s*(-?\d+\.?\d*)\s*$")
# The iteration separator the runner prints; also our block delimiter.
ITER_MARK = re.compile(r"Iteration time:")


def parse(path: str) -> dict[str, list[float]]:
    series: dict[str, list[float]] = defaultdict(list)
    with open(path, errors="replace") as fh:
        for line in fh:
            m = METRIC.match(line)
            if not m:
                continue
            key, value = m.group(1).strip(), float(m.group(2))
            series[key].append(value)
    return dict(series)


def trend(values: list[float], head_frac: float = 0.3) -> tuple[float, float]:
    """Compare two RECENT windows, skipping the opening transient.

    Comparing the last slice against the very first one is how this tool
    reported a flat objective as "rising": every series climbs off its
    initialisation, so including iteration 0 makes almost anything look
    healthy. Judging the second half against the third quarter asks the
    question that matters -- is it STILL improving -- rather than "is it
    better than random".
    """
    if len(values) < 8:
        return (values[0] if values else 0.0, values[-1] if values else 0.0)
    mid = len(values) // 2
    k = max(2, int(len(values) * head_frac))
    early = values[mid : mid + k] or values[mid:]
    late = values[-k:]
    return sum(early) / len(early), sum(late) / len(late)


def find(series: dict[str, list[float]], *fragments: str) -> dict[str, list[float]]:
    return {
        k: v
        for k, v in series.items()
        if any(f.lower() in k.lower() for f in fragments)
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--min-iters", type=int, default=200,
                   help="refuse to judge before this many logged iterations")
    p.add_argument("--objective", default="beat_bob",
                   help="the stage-1 positive term that must be rising")
    args = p.parse_args()

    series = parse(args.log)
    if not series:
        print("VERDICT: UNKNOWN — no metrics parsed yet (is the run past setup?)")
        return 2

    iters = max(len(v) for v in series.values())
    fails: list[str] = []
    warns: list[str] = []
    notes: list[str] = []

    # --- FAIL: penalty terms must never log positive -------------------------
    for key, values in find(series, "penalty", "_l1", "cost").items():
        if not key.startswith("Episode_Reward"):
            continue
        worst = max(values)
        if worst > 1e-6:
            fails.append(f"{key} logged +{worst:.4f} — sign is inverted; the "
                         f"policy is being PAID to violate it")

    # --- FAIL: NaN physics ---------------------------------------------------
    for key, values in find(series, "nan").items():
        if max(values) > 0:
            fails.append(f"{key} fired ({max(values):.2f}) — physics diverged")

    # --- WARN: the stage-1 objective must be climbing ------------------------
    obj = find(series, args.objective)
    if not obj:
        warns.append(f"no series matching '{args.objective}' — is it weighted 0?")
    for key, values in obj.items():
        first, last = trend(values)
        notes.append(f"{key}: {first:.3f} -> {last:.3f}")
        if last <= first * 1.02:
            warns.append(f"{key} has PLATEAUED ({first:.3f} -> {last:.3f}) — the "
                         f"primary objective has stopped improving. If it is also "
                         f"far from its 1.0 ceiling, the term is probably "
                         f"satisfiable without doing the task.")

    # --- WARN: falls should be decreasing ------------------------------------
    for key, values in find(series, "fell_over").items():
        first, last = trend(values)
        notes.append(f"{key}: {first:.2f} -> {last:.2f}")
        if last > first:
            warns.append(f"{key} rising ({first:.2f} -> {last:.2f}) — losing stability")

    # --- WARN: the duck is moving LESS over time -----------------------------
    # A rhythmic task satisfied by standing still is the failure mode a
    # tracking Gaussian invites: holding the mean pose collects partial credit
    # on every step while moving risks a fall that zeroes the gate.
    for key, values in find(series, "peak_height").items():
        first, last = trend(values)
        notes.append(f"{key}: {first:.4f} -> {last:.4f}")
        if last < first * 0.9:
            warns.append(f"{key} shrinking ({first:.4f} -> {last:.4f}) — the policy "
                         f"is learning to move LESS; a motion reward that pays for "
                         f"stillness is the usual cause")

    # --- WARN: entropy collapse ----------------------------------------------
    for key, values in find(series, "entropy").items():
        first, last = trend(values)
        notes.append(f"{key}: {first:.3f} -> {last:.3f}")
        if first > 0 and last < first * 0.25:
            warns.append(f"{key} collapsed ({first:.2f} -> {last:.2f}) — policy "
                         f"went degenerate; the reward curve can still look fine")

    for key, values in find(series, "Mean reward", "Mean episode length").items():
        first, last = trend(values)
        notes.append(f"{key}: {first:.2f} -> {last:.2f}")

    print(f"Parsed {iters} logged iterations from {args.log}\n")
    if notes:
        print("Signals:")
        for n in notes:
            print(f"  {n}")
        print()
    for f in fails:
        print(f"  FAIL  {f}")
    for w in warns:
        print(f"  WARN  {w}")
    if not fails and not warns:
        print("  All conditions clean.")
    print()

    if iters < args.min_iters:
        print(f"VERDICT: TOO EARLY — {iters} < {args.min_iters} iterations. "
              f"Keep it running; nothing here is trustworthy yet.")
        return 2
    if fails:
        print("VERDICT: ABORT — a listed FAIL invalidates the run. Fix and relaunch.")
        return 1
    if warns:
        print("VERDICT: INVESTIGATE — no fatal fault, but the run is not on track.")
        return 3
    print("VERDICT: CONTINUE — the reward stack is behaving. Let it run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

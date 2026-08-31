#!/usr/bin/env python3
"""Measure this GPU's real training throughput, and size the curriculum to it.

Run this FIRST on a new node. Everything downstream -- how many iterations fit
in a budget, whether the curriculum stages need compressing, whether to buy one
node or eight -- follows from one number this produces and nothing else can.

Method: two timed runs per environment count, at different iteration counts.
The marginal cost is ``(t_long - t_short) / (iters_long - iters_short)``, which
cancels the fixed startup -- import, MJCF parse, Warp kernel compile -- that
would otherwise dominate a short benchmark and make big batches look bad.

    python scripts/calibrate_throughput.py --task Mjlab-BeatDance-Flat-MicroDuck \
        --envs 1024 2048 4096 8192 --budget-minutes 45
"""

from __future__ import annotations

import argparse
import subprocess
import time

# Rollout length upstream trains with; one iteration = envs * this many steps.
STEPS_PER_ENV = 24


def time_run(task: str, num_envs: int, iters: int, extra: list[str]) -> float:
    cmd = [
        "uv", "run", "train", task,
        "--env.scene.num-envs", str(num_envs),
        "--agent.max_iterations", str(iters),
        *extra,
    ]
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"train failed at {num_envs} envs / {iters} iters:\n{tail}")
    return elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Mjlab-BeatDance-Flat-MicroDuck")
    p.add_argument("--envs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--short", type=int, default=10, help="iterations, short run")
    p.add_argument("--long", type=int, default=30, help="iterations, long run")
    p.add_argument("--budget-minutes", type=float, default=45.0)
    p.add_argument("--curriculum-iters", type=int, default=3400,
                   help="iterations the curriculum is written for")
    p.add_argument("--reference-envs", type=int, default=4096,
                   help="batch size the PPO hyperparameters are tuned at; the "
                        "curriculum is sized against this row, not the fastest")
    p.add_argument("extra", nargs="*", help="extra flags forwarded to train")
    args = p.parse_args()

    if args.long <= args.short:
        raise SystemExit("--long must exceed --short")

    print(f"{'envs':>7} {'startup s':>10} {'s/iter':>8} {'steps/s':>12} {'iters/budget':>13}")
    print("-" * 54)
    rows: list[tuple[int, float, float, int]] = []
    for n in args.envs:
        t_short = time_run(args.task, n, args.short, args.extra)
        t_long = time_run(args.task, n, args.long, args.extra)
        per_iter = (t_long - t_short) / (args.long - args.short)
        if per_iter <= 0:
            print(f"{n:7d}  measurement noise exceeded the signal; raise --long")
            continue
        startup = t_short - per_iter * args.short
        steps_s = n * STEPS_PER_ENV / per_iter
        iters = int((args.budget_minutes * 60 - startup) / per_iter)
        print(f"{n:7d} {startup:10.1f} {per_iter:8.3f} {steps_s:12,.0f} {iters:13,d}")
        rows.append((n, steps_s, per_iter, iters))

    if not rows:
        return 1
    fastest = max(rows, key=lambda r: r[1])
    most_iters = max(rows, key=lambda r: r[3])
    print()
    print(f"Highest throughput: {fastest[0]} envs at {fastest[1]:,.0f} env-steps/s.")
    print(f"Most iterations in budget: {most_iters[0]} envs at ~{most_iters[3]:,}.")
    if fastest[0] != most_iters[0]:
        print(
            "These disagree, and the disagreement is the point: a bigger batch buys\n"
            "more SAMPLES per iteration but fewer ITERATIONS in a fixed wall-clock\n"
            "budget -- and a curriculum is scheduled in iterations. Prefer the batch\n"
            "the PPO hyperparameters were tuned at (upstream: 4096) unless you have a\n"
            "reason to move, and take the iteration count from that row."
        )
    # Size the curriculum against the reference batch when it was measured.
    reference = next((r for r in rows if r[0] == args.reference_envs), fastest)
    n, steps_s, per_iter, iters = reference
    scale = iters / args.curriculum_iters
    print()
    print(f"At the reference batch of {n} envs: {steps_s:,.0f} env-steps/s, "
          f"~{iters:,} iterations in {args.budget_minutes:.0f} min.")
    print(f"A full {args.curriculum_iters}-iteration run there takes "
          f"~{args.curriculum_iters * per_iter / 60:.0f} min.")
    if scale >= 1.0:
        print(f"The {args.curriculum_iters}-iteration curriculum fits ({scale:.1f}x headroom). Run it as written.")
    else:
        print(
            f"The {args.curriculum_iters}-iteration curriculum does NOT fit. Scale the stage\n"
            f"boundaries in beat_dance_env_cfg.py by ~{scale:.2f} so every reward term still\n"
            f"activates -- an uncompressed schedule would spend the whole budget on the bob\n"
            f"and never switch the footfall term on at all."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

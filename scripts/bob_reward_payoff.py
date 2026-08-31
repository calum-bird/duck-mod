#!/usr/bin/env python3
"""How much does the bob reward pay a duck that does NOTHING?

Run 1 (2026-08-31) learned to stand rigidly still. This is the arithmetic that
made that the rational choice, and the tool for checking that a proposed std
actually fixes it before spending an hour of H100 to find out.

A Gaussian tracking reward against a moving reference is always partly
satisfiable by holding the reference's mean: the error is then just the
reference's own excursion, and if that excursion is small relative to std, the
free score is high. The policy compares that free score against tracking --
which costs effort and risks a fall that zeroes the hard gate -- and picks
correctly. The fix is to make the free score small, which means std must be
small relative to the AMPLITUDE, not merely "the error we care about".

    python scripts/bob_reward_payoff.py
"""

from __future__ import annotations

import argparse
from microduck_dance.beat_clock import bob_payoff as payoff


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amps", type=float, nargs="+", default=[0.008, 0.014, 0.020])
    p.add_argument("--stds", type=float, nargs="+", default=[0.010, 0.008, 0.006, 0.004])
    p.add_argument("--lag", type=float, default=0.003, help="rms tracking error of a good policy (m)")
    p.add_argument("--min-ratio", type=float, default=3.0,
                   help="tracking must out-earn standing still by at least this")
    args = p.parse_args()

    print(f"{'amp mm':>7} {'std mm':>7} {'stand-still':>12} {'good track':>11} {'ratio':>7}  verdict")
    print("-" * 62)
    for amp in args.amps:
        for std in args.stds:
            still, track = payoff(amp, std, args.lag)
            ratio = track / max(still, 1e-9)
            ok = "ok" if ratio >= args.min_ratio else "STILLNESS PAYS"
            print(f"{amp * 1000:7.0f} {std * 1000:7.0f} {still:12.3f} {track:11.3f} {ratio:7.2f}  {ok}")
        print()
    print(f"Rule of thumb: keep the ratio above {args.min_ratio:.0f}x. Run 1 shipped 14 mm / 10 mm")
    print("(ratio 1.94) and the policy correctly chose to stand still.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

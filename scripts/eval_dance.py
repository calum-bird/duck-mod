#!/usr/bin/env python3
"""Headless eval battery: run the policy at several tempi, measure the dance.

    MICRODUCK_RL=~/work/microduck_rl python scripts/eval_dance.py \
        --policy beat_dance.onnx --bpm 60 90 120 140 160 --seconds 20

No display, no viewer, no GPU -- one duck stepped at 50 Hz is a CPU workload.
Prints the tempo-sweep table from ``microduck_dance.eval`` and, with --json,
writes the flat summary for wandb or a mission chart. This is the artifact to
compare between reward variants: "on-beat 78% at 120, 31% at 160" is a finding;
"it looks better" is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--microduck-rl", default=None, help="checkout path (or set MICRODUCK_RL)")
    p.add_argument("--bpm", type=float, nargs="+", default=[60, 90, 120, 140, 160])
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--bob", type=float, default=0.018)
    p.add_argument("--sway", type=float, default=0.014)
    p.add_argument("--yaw", type=float, default=0.15)
    p.add_argument("--scene", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--save-trajectories", default=None, help="directory to save raw rollouts")
    args = p.parse_args()

    import torch

    from microduck_dance import eval as dance_eval
    from microduck_dance.dance_driver import BeatController
    from microduck_dance.headless import rollout

    metrics = []
    for bpm in args.bpm:
        controller = BeatController(bpm=bpm, bob=args.bob, sway=args.sway, yaw=args.yaw)
        try:
            traj = rollout(
                policy_onnx=args.policy,
                microduck_rl=args.microduck_rl,
                controller=controller,
                seconds=args.seconds,
                scene=args.scene,
            )
        except Exception as exc:  # a fall or a load failure should not lose the sweep
            print(f"[warn] {bpm:.0f} BPM rollout failed: {exc}")
            continue
        if args.save_trajectories:
            out = Path(args.save_trajectories)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(traj, out / f"traj_{bpm:.0f}.pt")
        metrics.append(dance_eval.evaluate(traj))

    if not metrics:
        raise SystemExit("no rollouts completed")

    print(dance_eval.format_report(metrics))
    if args.json:
        Path(args.json).write_text(json.dumps(dance_eval.summarize(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

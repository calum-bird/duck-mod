#!/usr/bin/env python3
"""Headless eval battery: run the policy at several tempi, measure the dance.

    MUJOCO_GL=egl MICRODUCK_RL=~/microduck_rl \
        python scripts/eval_dance.py --policy beat_dance.onnx \
            --bpm 60 90 120 140 160 --seconds 20

Prints the tempo-sweep table from ``microduck_dance.eval`` and, with --json,
writes the flat summary for wandb or a mission chart. This is the artifact to
compare between reward variants -- "on-beat 78% at 120, 31% at 160" is a
finding; "it looks better" is not.

One tempo per subprocess: upstream's viewer loop owns the process, so the
cleanest way to run five of them is five runs. --bpm with several values
re-execs this script once per tempo and aggregates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_one(args, bpm: float, out: Path) -> None:
    """Drive one tempo to completion and save the recorded trajectory."""
    import torch

    from microduck_dance.dance_driver import (
        BeatController,
        TrajectoryRecorder,
        load_infer_policy,
        make_dance_inference,
    )

    infer = load_infer_policy(args.microduck_rl)
    controller = BeatController(bpm=bpm, bob=args.bob, sway=args.sway, yaw=args.yaw)
    recorder = TrajectoryRecorder(bpm=bpm, dt=1.0 / args.rate, bob=args.bob, sway=args.sway)

    limit = int(args.seconds * args.rate)

    base = make_dance_inference(infer.PolicyInference, controller, recorder)

    class Bounded(base):  # type: ignore[valid-type,misc]
        def update_ground_pick_phase(self, dt: float) -> None:
            super().update_ground_pick_phase(dt)
            if len(recorder.bar_phase) >= limit:
                torch.save(recorder.to_trajectory(), out)
                # The viewer loop owns the process; stopping it from a frame
                # callback means unwinding out of it. The trajectory is already
                # on disk, so this loses nothing.
                raise SystemExit(0)

    infer.PolicyInference = Bounded
    sys.argv = ["infer_policy.py", "--walking", args.policy, "--new-cmd-obs"]
    try:
        infer.main()
    except SystemExit:
        pass
    if not out.exists():
        raise SystemExit(f"Run ended before {args.seconds}s at {bpm} BPM — policy fell?")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--microduck-rl", default=None)
    p.add_argument("--bpm", type=float, nargs="+", default=[60, 90, 120, 140, 160])
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--bob", type=float, default=0.018)
    p.add_argument("--sway", type=float, default=0.014)
    p.add_argument("--yaw", type=float, default=0.15)
    p.add_argument("--outdir", default="eval_out")
    p.add_argument("--json", default=None)
    p.add_argument("--_single", type=float, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args._single is not None:
        run_one(args, args._single, outdir / f"traj_{args._single:.0f}.pt")
        return 0

    for bpm in args.bpm:
        out = outdir / f"traj_{bpm:.0f}.pt"
        if out.exists():
            out.unlink()
        cmd = [sys.executable, __file__, "--policy", args.policy,
               "--seconds", str(args.seconds), "--rate", str(args.rate),
               "--bob", str(args.bob), "--sway", str(args.sway), "--yaw", str(args.yaw),
               "--outdir", str(outdir), "--_single", str(bpm)]
        if args.microduck_rl:
            cmd += ["--microduck-rl", args.microduck_rl]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[warn] {bpm} BPM run failed (exit {result.returncode})")

    import torch

    from microduck_dance import eval as dance_eval

    metrics = []
    for bpm in args.bpm:
        out = outdir / f"traj_{bpm:.0f}.pt"
        if not out.exists():
            print(f"[warn] no trajectory for {bpm} BPM")
            continue
        metrics.append(dance_eval.evaluate(torch.load(out, weights_only=False)))

    if not metrics:
        raise SystemExit("no trajectories recorded")
    print(dance_eval.format_report(metrics))
    if args.json:
        Path(args.json).write_text(json.dumps(dance_eval.summarize(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

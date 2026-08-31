#!/usr/bin/env python3
"""Watch a trained dance policy in the MuJoCo viewer, with a live tempo.

    MICRODUCK_RL=~/microduck_rl python scripts/play_dance.py \
        --policy beat_dance.onnx --bpm 128

    # sweep 60 -> 160 BPM over 90 s: shows exactly where tempo lock lets go
    ... --policy beat_dance.onnx --bpm 60 --sweep-to 160 --sweep-seconds 90

    # live control from another shell
    ... --policy beat_dance.onnx --bpm-file /tmp/dance_bpm
    echo 140 > /tmp/dance_bpm

The dance policy is loaded into upstream's always-on "walking" slot and
--new-cmd-obs is forced, because this task uses the 13-wide command layout.
Upstream's own key bindings still work for the head and the camera.
"""

from __future__ import annotations

import argparse
import sys

from microduck_dance.dance_driver import (
    BeatController,
    load_infer_policy,
    make_dance_inference,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="dance policy ONNX")
    p.add_argument("--microduck-rl", default=None, help="microduck_rl checkout (or set MICRODUCK_RL)")
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--sweep-to", type=float, default=None, help="ramp to this BPM")
    p.add_argument("--sweep-seconds", type=float, default=60.0)
    p.add_argument("--bpm-file", default=None, help="poll this file for live tempo")
    p.add_argument("--bob", type=float, default=0.018, help="bob amplitude (m)")
    p.add_argument("--sway", type=float, default=0.014, help="sway amplitude (m)")
    p.add_argument("--yaw", type=float, default=0.15, help="twist amplitude (rad)")
    return p


def main() -> int:
    args, passthrough = build_parser().parse_known_args()

    infer = load_infer_policy(args.microduck_rl)
    controller = BeatController(
        bpm=args.bpm,
        bob=args.bob,
        sway=args.sway,
        yaw=args.yaw,
        sweep_to=args.sweep_to,
        sweep_seconds=args.sweep_seconds,
        bpm_file=args.bpm_file,
    )
    # main() looks the class up by module global, so rebinding installs ours.
    infer.PolicyInference = make_dance_inference(infer.PolicyInference, controller)

    sys.argv = [
        "infer_policy.py",
        "--walking", args.policy,
        "--new-cmd-obs",
        *passthrough,
    ]
    return infer.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())

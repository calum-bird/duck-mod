#!/usr/bin/env python3
"""Render a dance rollout to video, offscreen, beat clock locked to a track.

    MICRODUCK_RL=~/work/microduck_rl MUJOCO_GL=egl \
        python scripts/render_dance.py --policy beat_dance.onnx \
            --bpm 120 --seconds 16 --out dance.mp4 [--audio song.mp3 --audio-offset 0.0]

Frames come from MuJoCo's offscreen renderer (EGL on a headless box), video is
encoded at exactly the control rate so one control step = one frame and the
clock cannot drift against the music, and when --audio is given the track is
muxed in with ffmpeg at the end, offset so beat 1 of the CLOCK lands on
--audio-offset seconds into the file (line the offset up with the track's
first downbeat and the duck is dancing to the song, not near it).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from microduck_dance.dance_driver import (
    BeatController,
    load_infer_policy,
    make_dance_inference,
)
from microduck_dance.headless import DECIMATION, MICRODUCK_SCENE, condition_model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--microduck-rl", default=None)
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--seconds", type=float, default=16.0)
    p.add_argument("--bob", type=float, default=0.014)
    p.add_argument("--sway", type=float, default=0.012)
    p.add_argument("--yaw", type=float, default=0.15)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="dance.mp4")
    p.add_argument("--audio", default=None, help="track to mux (mp3/wav)")
    p.add_argument("--audio-offset", type=float, default=0.0,
                   help="seconds into the audio where its first downbeat falls")
    args = p.parse_args()

    import imageio_ffmpeg
    import mujoco
    import numpy as np

    infer = load_infer_policy(args.microduck_rl)
    root = Path(args.microduck_rl or os.environ["MICRODUCK_RL"])
    model = mujoco.MjModel.from_xml_path(str(root / MICRODUCK_SCENE))
    condition_model(model)
    data = mujoco.MjData(model)
    # The XML's offscreen framebuffer defaults to 640x480; asking the renderer
    # for more raises the <visual><global offwidth/> error. Grow it in the
    # loaded model instead of editing the scene.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    control_dt = DECIMATION * model.opt.timestep
    fps = 1.0 / control_dt

    controller = BeatController(bpm=args.bpm, bob=args.bob, sway=args.sway, yaw=args.yaw)
    cls = make_dance_inference(infer.PolicyInference, controller)
    policy = cls(model, data, walking_onnx_path=args.policy,
                 use_projected_gravity=True, new_cmd_obs=True)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 0.65, -12.0, 135.0

    raw = args.out + ".video.mp4"
    writer = imageio_ffmpeg.write_frames(
        raw, (args.width, args.height), fps=fps,
        codec="libx264", quality=8, pix_fmt_in="rgb24",
    )
    writer.send(None)

    steps = int(args.seconds / control_dt)
    for i in range(steps):
        policy.update_ground_pick_phase(control_dt)
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        # Track the trunk with a little smoothing so footwork reads clearly
        # without the camera pumping on every bob.
        cam.lookat[:] = 0.9 * np.array(cam.lookat) + 0.1 * data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        writer.send(np.ascontiguousarray(renderer.render()))
        if i % int(fps * 4) == 0:
            print(f"  {i / fps:5.1f}s / {args.seconds:.0f}s", flush=True)
    writer.close()

    if args.audio:
        # Shift the track so its first downbeat hits t=0, where the clock's
        # beat 1 is; pad video length wins (-shortest).
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", raw, "-ss", str(args.audio_offset), "-i", args.audio,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
            args.out,
        ], check=True)
        os.remove(raw)
    else:
        os.replace(raw, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

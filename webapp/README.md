# Microduck Dance Floor

The [pollen-robotics/microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
playground (MuJoCo WebAssembly physics + onnxruntime-web policies, all
in-browser, no backend) with one addition: **drop in any song and the duck
dances to it.**

```bash
cd webapp
npm install
npm run dev     # open the printed localhost URL
```

With **bun**, force bun's own runtime — `bun run dev` alone follows Vite's
`node` shebang to your system Node, and Vite 8's rolldown needs Node ≥ 20.12
(`util.styleText`), which older Nodes lack:

```bash
bun install
bun --bun run dev
```

Click **♪ PLAY A SONG** (bottom-left) and pick an mp3/wav/m4a/ogg. The app:

1. decodes it with WebAudio and measures its tempo + first downbeat
   in-browser (spectral-flux onset envelope + autocorrelation — the same
   algorithm as `scripts/beat_track.py`),
2. folds an out-of-band tempo into the policy's trained 100–140 BPM band by
   powers of two (only 2^k ratios keep the duck on the song's grid),
3. plays the track and drives the beat-dance policy
   (`public/policies/beat_dance_head.onnx`, the duck-mod fine-tune) with a
   clock derived from `AudioContext.currentTime` — sync is against the same
   hardware clock your ears hear, so it cannot drift.

**You can walk around mid-song**: WASD/arrows instantly hand the duck to the
walking policy, and half a second of idle hands it back to the dance. The
frozen 61-dim observation contract shared by every Microduck policy is what
makes the mid-track hot-swap safe. Falls during a dance go through the same
automatic get-up recovery as walking, then resume dancing.

Everything else from the playground still works: sit, kick, roll, ground
pick, head mode, the roller variant, colourways, gamepad and touch.

Tips:

- Steady, produced tracks work best; the analyzer warns on live-drum /
  rubato pulses (the clock is fixed-tempo — a drifting song drifts off it).
- The trained sweet spot is 100–140 BPM; outside it the fold note in the
  dock says what tempo the duck is actually dancing.

## Provenance

Forked from the `app/` of
[pollen-robotics/microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
(robot model, meshes and base policies from
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) and
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)).
Additions here: `src/game/dance.js` (beat analysis + audio-clock player),
`src/ui/MusicDock.jsx`, the `dance` mode wiring in `src/game/game.js`, and
the fine-tuned `beat_dance_head.onnx` policy trained in this repo.

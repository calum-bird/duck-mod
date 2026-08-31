# Running this on a givemeanode H100

## What's available

Only one GPU shape: **H100**, as `h100-1` (one GPU) or `h100-8` (a full
8-GPU machine). Plus CPU-only `cpu-2` and `cpu-8`. No A100/L40S/4090 tier.

At the H100 list rate the per-session minimum works out to **~$3.60/GPU-hour**
(the quoted minimums are $0.90 for 15 min on a 1x, $9.60 for 20 min on an 8x).

The limits that shape the plan:

| Limit | Value | Why it matters |
|---|---|---|
| `max_nodes` | 64 | breadth is available — run reward variants in parallel |
| `max_running_job_gpus_per_org` | 64 | same, via `submit_jobs` |
| `max_sweep_variants` | 256 | one call queues a whole grid |
| `max_snapshots` | 16 (250 GiB free) | bake the env once, fan out clones |
| grace window | 15 min on 1x | don't churn nodes; a stop/start cycle costs $0.90 |

Because only H100s exist here, "cheaper mid-tier GPUs" isn't an option — but
the *strategy* survives intact, since 64 concurrent nodes means parallelism is
still the thing to buy. Reward design is a hypothesis search; six 30-minute
runs in parallel beat one run that is 20% faster.

## Image

**`pytorch-2.13-robotics-cuda12.9`.** Not for its torch — `uv sync` installs
microduck_rl's own pinned `torch==2.9.1` into the venv regardless — but for the
system layer underneath: ffmpeg, OpenCV, and **headless EGL/Vulkan**. That EGL
is what makes `MUJOCO_GL=egl` rendering work on a machine with no display,
which is what `scripts/eval_dance.py` and any recorded video depend on. Without
it the first render call fails several billed minutes into a run.

Not the JAX image: microduck_rl is **mjlab + MuJoCo Warp on PyTorch**, not MJX.
`jax-0.11-cuda12.9` is the right choice for an MJX rollout farm and the wrong
one for this stack.

Isaac Sim doesn't run on H100 anywhere (no RT cores). Irrelevant here — MuJoCo
is the whole pipeline.

One thing to confirm in the first ten minutes: torch 2.9.1's CUDA wheels and
Warp's kernel compilation against a **CUDA 12.9** toolkit. Expected to be fine
(drivers are backward compatible), and the smoke test settles it before any
long run starts.

## The run plan

**Stage 0 — calibrate. Done: 2026-08-31, 12.8 min, $0.85.** Measured on one
`h100-1` (H100 80GB HBM3), `Mjlab-BeatDance-Flat-MicroDuck`, two timed runs per
env count so fixed startup cancels out of the marginal cost:

| envs | startup s | s/iter | env-steps/s | iters in 45 min |
|---|---|---|---|---|
| 1024 | 14.3 | 1.142 | 21,511 | 2,350 |
| 2048 | 15.1 | 1.226 | 40,093 | 2,190 |
| 4096 | 17.5 | 1.418 | 69,339 | 1,892 |
| 8192 | 25.1 | 2.058 | 95,542 | 1,299 |

**Operating point: 4096 envs.** Throughput keeps rising to 8192, but the gain
per doubling is flattening (1.86x, 1.73x, 1.38x) and *iterations* fall as the
batch grows — and the curriculum counts iterations, not samples. 4096 is also
upstream's default, so the PPO hyperparameters are tuned there.

**The 3400-iteration curriculum needs ~80 min at 4096 (~$5.35), not 45.** Either
budget the 80 minutes or scale the stage boundaries by ~0.56. Running it
uncompressed in 45 minutes would spend the whole budget on the bob and never
switch the footfall term on — which is exactly the failure calibration exists to
prevent.

**Then snapshot the node** (done: `snap-dm756`). `from_snapshot` restores the volume *and* the image
it was built under, so venvs and compiled extensions resume intact. Every
subsequent node skips the ~3 GB `uv sync` entirely. This is the single highest-
leverage setup step: without it, a 30-minute experiment pays 20% overhead just
to exist.

**Stage 1 — one honest run (~1 h, ~$4).** Compress the curriculum by the factor
calibration reports, then train. Expect a bob locked to 120 BPM, probably sway;
footfall timing is a coin-flip; tempo generalisation almost certainly not.

**Stage 2 — the search (parallel).** Fan out from the snapshot: one node per
reward variant, three seeds each. `scripts/eval_dance.py` gives every run the
same tempo-sweep table, which is what makes them comparable at all.

## Secrets

Pass credentials through `run_command`'s `env`, never in the command string.

- **wandb** — required for training as shipped. rsl_rl's runner defaults to the
  wandb logger and raises `UsageError: No API key configured` before the first
  iteration. `WANDB_MODE=disabled` makes it a no-op for unattended runs (that is
  how calibration ran). Export is genuinely independent: `export.py` accepts a
  local `checkpoint_file`. What a real key buys is run *comparison*, which is the
  entire point of Stage 2.
- **Hugging Face — not needed.** All 58 meshes and every robot XML are vendored
  in microduck_rl; `huggingface_hub` is imported only by its HF-Jobs submitter
  and result uploader. A token is needed only to submit to HF Jobs (we're using
  givemeanode instead) or to publish a finished policy to the Hub for the robot
  runtime to install — a deployment step, not a training one.

## Cost sketch

Measured rate: **$0.0666/min = $4.00/GPU-hour**, 15-minute session minimum.

| | GPU-hours | Cost |
|---|---|---|
| Calibrate + snapshot | 0.21 | **$0.85 (actual)** |
| One full 3400-iteration run @ 4096 | 1.34 | ~$5.35 |
| 6 variants x 3 seeds | 24 | ~$96 |
| A two-week search, ~8 GPU-h/day | ~110 | ~$440 |

Calibration cost 85 cents and turned "one run fits in an hour" into "one run is
80 minutes" — a 78% underestimate that would have produced a duck that bobs and
never steps.

## Housekeeping

`stop_node` the moment work finishes — a node bills while running or idling its
grace window. Use `hold_node` to keep one awake across turns rather than a sleep
loop. Group everything under one mission so cost and conclusions land on a page
a human can read.

## Gotchas found the hard way

- The container shell is `sh`, not bash: no `time` builtin. Time with
  `S=$(date +%s)` … `$(( $(date +%s) - S ))`.
- Piping a training run through `tail` buffers everything until exit, so a
  detached command shows an empty log for minutes. Redirect to a file instead.
- Warp compiles kernels on the first run (~2-4 min, CPU-bound, GPU at 0%). It
  caches to `~/.cache/warp`, after which startup is ~15 s. The snapshot carries
  that cache.
- The task package is installed into microduck_rl's own venv with
  `uv pip install --no-deps -e ~/work/duck-mod`, which avoids resolving a second
  ~4 GB dependency tree.
- A node wake from stopped was quoted at ~7.5-10 min. Restoring from the
  snapshot is the faster path back.

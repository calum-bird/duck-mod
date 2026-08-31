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

**Stage 0 — calibrate (~15 min, ~$1).** One `h100-1`. `scripts/setup_node.sh`,
then `scripts/calibrate_throughput.py`. This produces the one number everything
else depends on: env-steps/s, and therefore how many iterations fit a budget
and whether the curriculum needs compressing. It uses a two-point timing method
so fixed startup — imports, MJCF parse, Warp kernel compile — cancels out
instead of poisoning the measurement.

**Then snapshot the node.** `from_snapshot` restores the volume *and* the image
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

- **wandb** — wanted, not required. `export.py` accepts a local
  `checkpoint_file`, so train → export → play works without it. What wandb buys
  is run *comparison*, which is the entire point of Stage 2.
- **Hugging Face — not needed.** All 58 meshes and every robot XML are vendored
  in microduck_rl; `huggingface_hub` is imported only by its HF-Jobs submitter
  and result uploader. A token is needed only to submit to HF Jobs (we're using
  givemeanode instead) or to publish a finished policy to the Hub for the robot
  runtime to install — a deployment step, not a training one.

## Cost sketch

| | GPU-hours | ~Cost |
|---|---|---|
| Calibrate + snapshot | 0.3 | ~$1 |
| One full training run | 1 | ~$4 |
| 6 variants x 3 seeds x 1 h | 18 | ~$65 |
| A two-week search, ~8 GPU-h/day | ~110 | ~$400 |

Calibration costs about a dollar and tells you whether the honest number for
the rest is $65 or $650.

## Housekeeping

`stop_node` the moment work finishes — a node bills while running or idling its
grace window. Use `hold_node` to keep one awake across turns rather than a sleep
loop. Group everything under one mission so cost and conclusions land on a page
a human can read.

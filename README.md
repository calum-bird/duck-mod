# microduck-dance

A beat-synced dance skill for the [Microduck](https://github.com/pollen-robotics/microduck) —
Hugging Face / Pollen Robotics' $399 open-source bipedal duck.

The duck dances **in place** to an external beat: bobbing once per beat,
shifting its weight across each bar, twisting into the shift, and landing a
foot on the downbeat — at any tempo from 60 to 160 BPM, following a clock the
host sends at run time.

This is a **layer on top of** [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl),
not a fork of it. It imports the upstream training stack and registers two
extra mjlab tasks, so upstream fixes to the robot model, the BAM actuator model
and the domain randomisation flow straight through.

## Quickstart

```bash
uv sync
uv run list-envs | grep Dance

uv run train Mjlab-BeatDance-Flat-MicroDuck --env.scene.num-envs 4096
# ...or, with no local GPU:
uv run train Mjlab-BeatDance-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

Needs a CUDA GPU (or HF Jobs) and `uv`. Roughly 1–2 hours to something that
dances; longer to something that dances well.

**[docs/training.md](docs/training.md) is the real documentation** — the reward
design and why each term is shaped the way it is, the curriculum and its
ordering, what to watch in wandb, the known failure modes, and the sim-to-real
path.

## Tasks

| Task | Notes |
|---|---|
| `Mjlab-BeatDance-Flat-MicroDuck` | the task |
| `Mjlab-BeatDance-Flat-Backlash-MicroDuck` | ±1° servo gear play; **train this one before deploying to hardware** |

## Layout

```
src/microduck_dance/
├── beat_clock.py          # phase/tempo kernels — pure torch, no mjlab
├── commands.py            # BeatClockCommand: the training-time clock
├── mdp.py                 # dance rewards, penalties, curricula
├── beat_dance_env_cfg.py  # the env config and its curriculum
├── beat_source.py         # host-side clock for deployment — no dependencies
└── tasks/__init__.py      # mjlab task registration (entry point target)
tests/                     # reward/kernel tests run on CPU; cfg tests need mjlab
docs/training.md           # the recipe
```

## Tests

```bash
uv run --with pytest pytest tests/
```

Most of the suite runs on a plain CPU box with only `torch` — the reward terms
touch a narrow enough slice of the mjlab API to be driven by a stub env, so the
timing logic, the anti-exploit gates and the EMA behaviour are all testable
without a simulator. The config-level tests skip automatically when mjlab is
not installed.

## The one constraint to know

The runtime hot-swaps policies behind a **fixed 61-dimensional observation**
(48 proprioception + `[twist(3), head_pose(4), body_pose(6)]`). A new command
cannot be appended, so this task repurposes existing slots: `twist` carries the
beat clock, and three dims of `body_pose` carry the choreography amplitudes.
Any change here that alters the observation width makes the policy
undeployable alongside the walking one. There is a test guarding it.

## Licence

Apache-2.0, matching upstream.

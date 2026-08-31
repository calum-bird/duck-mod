# Training the Microduck to dance to a beat

This is the full recipe for the `Mjlab-BeatDance-Flat-MicroDuck` task: what it
trains, how to run it, what to watch while it runs, and how to get it onto the
robot.

## What the skill is

The duck dances **in place** to an external beat. It bobs once per beat, shifts
its weight left and right across each bar (two beats), twists into the shift,
and lands a foot on the downbeat — at any tempo from 60 to 160 BPM, following
whatever clock the host is sending, without wandering off its spot.

At deployment the host owns the tempo. The policy owns everything else.

## Prerequisites

* A CUDA GPU. Training is MuJoCo Warp (mjlab) + PPO at 4096 parallel
  environments; there is no practical CPU path.
* [`uv`](https://docs.astral.sh/uv/).
* No local GPU? Every `train` command below takes `--hf-jobs`, which submits
  the same run to Hugging Face Jobs instead.

## Install

```bash
uv sync                       # pulls mjlab-microduck from git, per pyproject.toml
uv run list-envs | grep Dance # confirm the tasks registered
```

Registration happens through the `mjlab.tasks` entry point, the same mechanism
upstream uses, so no wrapper script is needed. If the tasks do not appear, the
package is not installed into the environment you are running from.

Pin `mjlab-microduck` to a tag or commit in `pyproject.toml` before any run you
intend to reproduce — the default is `develop`, which moves.

## The four commands

```bash
# Train (1-2 h to a usable gait on one modern GPU, longer to a clean one)
uv run train Mjlab-BeatDance-Flat-MicroDuck --env.scene.num-envs 4096

# Watch it, at a fixed 120 BPM with every episode starting on a downbeat
uv run play Mjlab-BeatDance-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# Export to ONNX (this bakes in the observation normaliser — never hand-convert)
uv run scripts/export.py Mjlab-BeatDance-Flat-MicroDuck --wandb-run-path <...>

# Drive the exported policy in sim
uv run scripts/infer_policy.py --walking beat_dance.onnx
```

Resuming a run:

```bash
uv run train Mjlab-BeatDance-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

## Before you commit a GPU to it

```bash
uv run --with pytest pytest tests/          # config invariants + reward kernels
uv run train Mjlab-BeatDance-Flat-MicroDuck --env.scene.num-envs 64 \
    --agent.max_iterations 5                # smoke test
```

The smoke test catches the large majority of config errors — NaN-free stepping,
the 61-dim observation shape, a working export — for about a minute of GPU.

## How the beat reaches the policy

The runtime hot-swaps policies behind a **fixed 61-dimensional observation**:
48 proprioception values plus 13 command values laid out
`[twist(3), head_pose(4), body_pose(6)]`. A new command cannot be appended
without breaking every other policy on the robot, so this task **repurposes
existing slots** — the same thing the shipped ground-pick, sit-stand and spin
tasks do:

| Slot | Normal meaning | Here |
|---|---|---|
| `twist[0..1]` | linear x, y velocity | `cos(2πφ)`, `sin(2πφ)` — bar phase |
| `twist[2]` | yaw rate | tempo, normalised to [-1, 1] over 60–160 BPM |
| `head_pose[0..3]` | neck/head deltas | unchanged — a static attitude, held on average |
| `body_pose[1]` (y) | body y offset | **sway amplitude** (m) |
| `body_pose[2]` (z) | body z offset | **bob amplitude** (m) |
| `body_pose[5]` (yaw) | body yaw offset | **twist amplitude** (rad) |
| `body_pose[0,3,4]` | x, roll, pitch | zero-padded, slot kept alive |

Two details that matter:

* **Phase is sent as a (cos, sin) pair**, not a raw ramp. A sawtooth has a step
  discontinuity at every bar boundary that the policy would have to spend
  capacity learning to ignore.
* **One bar is two beats.** A biped shifts its weight left on one beat and
  right on the next, so the sway has a bar period while the bob has a beat
  period. A single-beat clock cannot express "which foot".

The amplitudes are commands rather than constants so the host can dial the
dance from subtle to exuberant at run time — including all the way to zero,
which is the "stand still on the beat" idle case the final curriculum stage
trains explicitly.

## The reward recipe

Removed from the inherited walking recipe: `track_linear_velocity` and
`track_angular_velocity` (the twist slot is no longer a velocity), `air_time`
(rewards long swings; the beat sets the cadence now) and `pose` (pulls the legs
to HOME, i.e. directly against the bob). `body_pose_tracking` is deleted rather
than zero-weighted, because that slot now means amplitudes and a future reader
raising its weight would get a 12 mm permanent trunk offset.

Added:

| Term | Weight | What it does |
|---|---|---|
| `beat_bob` | 4.0 | trunk height tracks a beat-locked sinusoid, lowest on the beat |
| `beat_sway` | → 2.0 | lateral offset tracks a bar-period sinusoid, measured along the trunk's own y axis |
| `beat_yaw` | → 1.0 | trunk twists into the sway |
| `beat_footfall` | → 40.0 | a foot strike, paid once per beat, scaled by timing accuracy |
| `foot_alternation_penalty` | → 20.0 | charges a strike that repeats the previous foot |
| `station_keeping_penalty` | 20.0 | charges time-averaged horizontal drift |
| `heading_drift_penalty` | 4.0 | charges time-averaged rotation |
| `stance_width_penalty` | -2.0 | keeps hip yaw/roll near HOME (cost-style → negative weight) |

Four design points worth knowing before you retune any of them:

**The footfall weight of 40 is not a typo.** It pays at most once per beat: at
120 BPM that is 2 payouts/s against a 50 Hz loop, so only 4% of steps pay
anything. At weight 40 and a typical timing accuracy of ~0.7 it is worth about
`40 × 0.04 × 0.7 ≈ 1.1` per step — the same order as `beat_bob`'s
`4.0 × ~0.8 = 3.2`. What PPO sees is reward *mass*, not weight. Give an impulse
term a dense term's weight and the policy will correctly ignore it.

**Everything positive is behind a hard gate.** `dance_gate` returns 0 unless
the duck is above 70 mm and within 25° of upright. Without it, the cheapest
rhythmic policy is to sit on its trunk and bounce to the beat — on beat,
scoring well, and terrible on real hardware. Soft nudge penalties do not close
that door.

**The drift penalties are EMAs, not instantaneous.** The sway *is* horizontal
displacement, so an instantaneous `|xy|` penalty would fight the sway reward
directly and net out to a duck standing rigidly still. Averaging over 4 s —
longer than the slowest bar, 2.0 s at 60 BPM — cancels the oscillation and
prices only the DC component, i.e. actually wandering off. Same trick upstream
uses against head droop.

**The head is deliberately unconstrained.** `head_pose_tracking` is set to
weight 0 and only the DC `head_pose_bias` term is active. Upstream measured
that an instantaneous head-tracking tax costs a walking policy ~0.77/step and
made it stop stepping altogether; the head is 38% of the robot's mass and
*must* oscillate. Here that oscillation is the point — the nod is half of what
makes this read as dancing — so the commanded attitude is enforced only on a
1 s average and the bob rides on top of it for free.

## The curriculum

Stages are phase-aligned, not simultaneous. Steps below are PPO iterations;
the config multiplies by 24 (`NUM_STEPS_PER_ENV`) because `common_step_counter`
counts environment steps.

| Iterations | What turns on |
|---|---|
| 0 – 800 | bob only, at a fixed ~120 BPM and mid-envelope amplitudes |
| 800 – 1600 | sway ramps in (1.0, then 2.0) |
| 1600 – 2000 | footfall timing ramps in at half weight |
| 2000 – 2600 | foot alternation ramps in |
| 2600 – 3400 | full footfall weight, tempo widens to 100–140, amplitudes widen |
| 3400+ | yaw twist, full 60–160 BPM range, mid-episode tempo changes, amplitudes reach zero |

The ordering is the part to preserve if you retune: asking for a foot on the
beat before the duck can stand rhythmically at all just suppresses stepping,
and showing it 60 and 160 BPM before it can hold 120 adds variance to a signal
it has not found yet. Every stage-0 weight must equal the term's initial weight
in the config — a mismatch silently moves the goalposts on the first curriculum
tick and shows up in wandb as a mysterious step change. There is a test for it.

## What to watch in wandb

* **Every `Episode_Reward/*_penalty` series must be ≤ 0.** A positive one means
  a sign got flipped, and the policy is now farming the violation. This is the
  single highest-value glance.
* `Episode_Reward/beat_bob` should climb first and plateau before iteration
  ~800. If it does not, nothing downstream will work — fix the bob before
  looking at anything else.
* `Episode_Reward/beat_footfall` climbing from its stage onset means strikes
  are landing near beats. Flat and near zero after ~500 iterations of being
  active usually means the bob is consuming all the leg travel; try lowering
  the bob amplitude range.
* A metric that steps **down** at a curriculum boundary means the pacing is
  wrong. Stretch the stage or delay the term rather than reweighting it.
* Before theorising about a failure, run a headless eval batch and measure it.
  "Falls off the beat 1 in 4 bars at 150 BPM" is actionable; "the rhythm looks
  off" is not.

## Known failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Duck stands rigidly still, high bob reward | bob amplitude near zero after the final curriculum stage | check the amplitude ranges actually widened; the zero-amplitude idle case is legitimate but should not dominate the batch |
| Bounces on its trunk in time | gate too permissive | raise `min_height` in `dance_gate` |
| Taps one foot, other planted | alternation weight too low or introduced too late | raise `ALTERNATION_WEIGHT`, or move its stage earlier |
| Drums several strikes per beat | should be impossible — the payout is rate-limited per beat | if seen, the nearest-beat accounting in `beat_footfall_reward` is broken; there are tests for it |
| Locks to 120 BPM, ignores others | tempo curriculum never widened, or the host's normalisation disagrees with training | `test_tempo_normalisation_matches_the_training_encoding` covers the second case |
| Slides across the floor | station-keeping EMA `tau_s` shorter than the bar period | keep `tau_s` ≥ 2× the slowest bar (≥ 4 s) |

## Sim-to-real

1. **Train the backlash twin** before deploying:
   `Mjlab-BeatDance-Flat-Backlash-MicroDuck`. Every task family has one, and
   for this skill it is not optional: the whole thing is timing, and ±1° of
   serial gear play is exactly what puts a real foot strike a few milliseconds
   off where the policy placed it.
2. **Keep the domain randomisation on.** It is inherited from the velocity env
   — BAM actuator friction, battery voltage sag, command delay, CoM, mass and
   inertia, IMU mounting error, encoder bias. The command-delay randomisation
   in particular is what makes the timing survive the real control path.
3. **Export with `scripts/export.py`.** It bakes the observation normaliser
   into the ONNX. In-sim play applies the normaliser either way and so hides
   normalisation bugs; a hand-converted checkpoint will look fine in `play` and
   fail on the robot.
4. **Feed the clock.** Use `microduck_dance.beat_source.BeatCommandSource` on
   the host — it is dependency-free and produces the same 13 values in the same
   slot order, with the same tempo normalisation (there is a test asserting the
   two encodings agree). Wire its output into your runtime's JSON-RPC command
   method; check the method name against your `microduck` runtime version
   rather than guessing.

   ```bash
   uv run beat-source --bpm 128 --seconds 2   # dry-run the clock, no robot needed
   ```

5. **Sync sparingly.** `BeatCommandSource.sync(0.0)` on a detected downbeat is
   for correcting drift, not for driving. The policy was trained on a phase
   that only ever advances smoothly; snapping every beat makes it jitter.

## Tuning levers

* **Timing strictness**: `DOWNBEAT_SIGMA` in `beat_clock.py` (default 0.08 of a
  beat ≈ 40 ms at 120 BPM, roughly human accuracy). Tighten for crisper
  placement, but tighten it late — a tight window early gives no gradient.
* **How big the dance is**: `MAX_BOB`, `MAX_SWAY`, `MAX_YAW` in the env cfg,
  plus the final stage of `dance_amplitude_range`.
* **An explicit head nod**: currently the nod emerges from the trunk bob and a
  free neck. If it reads as too subtle on hardware, add a term rewarding the
  neck pitch joint against a beat-locked reference — and expect to retune
  `head_pose_bias` alongside it, since the two pull on the same joints.
* **Which foot on which beat**: `beat_clock.stance_side` already reports which
  half of the bar we are in. Rewarding the matching foot is a stricter
  alternative to `foot_alternation_penalty`, at the cost of prescribing more of
  the gait than the policy needs to be told.

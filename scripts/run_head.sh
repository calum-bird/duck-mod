#!/usr/bin/env bash
# Head-swing fine-tune launch + gate. Resumes run 2's model_3800.pt into the
# Mjlab-BeatDanceHead task (flattened curriculum + head_yaw_power ramp).
#
# Resume mechanics (mjlab 1.3.0 scripts/train.py + utils/os.get_checkpoint_path):
#   --agent.resume True resolves logs/rsl_rl/<experiment_name>/<load_run regex>/
#   <load_checkpoint regex>. experiment_name "beat_dance_head" is a FRESH dir,
#   so the run-2 checkpoint is PLANTED under it first and targeted explicitly.
#   runner.load restores model + optimizer + iteration counter (logging resumes
#   at 3800); the env's common_step_counter starts at 0, which is exactly what
#   the head ramp is keyed to and why every other schedule was flattened.
#
# Cost: gate at ~500 iters (~12 min, ~$1); full 1200 iters ~28 min (~$1.90).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

RUN2_SHA=325e3410ba38fe18afb859dcbd2b4854bc4c5573473cd175ae9b5d7b32ee52d9
CANONICAL=~/work/microduck_rl/wandb/run-20260831_053914-e2d0qy6l/files/model_3800.pt

cd ~/work/duck-mod && git pull -q origin claude/microduck-training-27jcop
cd ~/work/microduck_rl
uv sync -q
uv pip install -q --no-deps -e ~/work/duck-mod
# Buffer list-envs to a file: grep -q on a live pipe under pipefail SIGPIPEs
# the producer and fails a passing check (aborted two run-2 launches).
uv run --no-sync list-envs > /tmp/envs.txt 2>&1 || true
grep -q BeatDanceHead /tmp/envs.txt || {
    uv pip install --force-reinstall --no-deps -e ~/work/duck-mod
    uv run --no-sync list-envs > /tmp/envs.txt 2>&1 || true
    grep -q BeatDanceHead /tmp/envs.txt || { echo "BeatDanceHead task missing"; exit 1; }
}

uv pip install -q pytest
uv run --no-sync python -m pytest ~/work/duck-mod/tests/ -q

# ---- Plant the run-2 checkpoint under the fresh experiment dir -------------
SEED_DIR=logs/rsl_rl/beat_dance_head/zz_run2_seed
mkdir -p "$SEED_DIR"
if [ ! -f "$SEED_DIR/model_3800.pt" ]; then
    SRC=""
    for c in "$CANONICAL" /tmp/serve/run2_model.pt; do
        [ -f "$c" ] && SRC="$c" && break
    done
    [ -n "$SRC" ] || { echo "run-2 checkpoint not found on this node"; exit 1; }
    cp "$SRC" "$SEED_DIR/model_3800.pt"
fi
echo "$RUN2_SHA  $SEED_DIR/model_3800.pt" | sha256sum -c - || {
    echo "checkpoint hash mismatch — wrong file, refusing to fine-tune it"; exit 1; }

# ---- Smoke: prove the checkpoint loads into the new task (~1 min, CPU-ish) -
WANDB_MODE=disabled uv run --no-sync train Mjlab-BeatDanceHead-Flat-MicroDuck \
    --env.scene.num-envs 512 --agent.max_iterations 3 \
    --agent.resume True --agent.load_run zz_run2_seed \
    --agent.load_checkpoint model_3800.pt > ~/smoke_head.log 2>&1 || {
    echo "SMOKE FAILED"; tail -40 ~/smoke_head.log; exit 1; }
grep -q "Loading model checkpoint from" ~/smoke_head.log || {
    echo "smoke ran but never loaded the checkpoint"; tail -40 ~/smoke_head.log; exit 1; }
echo "SMOKE_OK — checkpoint loads, task trains"
rm -rf logs/rsl_rl/beat_dance_head/2*  # scrub smoke run dirs so resume regexes stay clean

# ---- The real launch -------------------------------------------------------
# WANDB_API_KEY arrives via the command environment, never this file.
uv run --no-sync train Mjlab-BeatDanceHead-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 1200 \
    --agent.resume True --agent.load_run zz_run2_seed \
    --agent.load_checkpoint model_3800.pt \
    > ~/train_head.log 2>&1 &
TRAIN_PID=$!
echo "training pid $TRAIN_PID; gate at ~500 iterations"

iters_logged() {
    n=$(grep -c "Iteration time" ~/train_head.log 2>/dev/null || true)
    [ -n "$n" ] || n=0
    printf '%s' "$n"
}
wait_for_iters() {
    while [ "$(iters_logged)" -lt "$1" ]; do
        kill -0 $TRAIN_PID 2>/dev/null || { echo "trainer died at $(iters_logged) iterations"; tail -30 ~/train_head.log; exit 1; }
        sleep 30
    done
}

# ---- The gate --------------------------------------------------------------
# The ramp holds head_yaw_power's weight at 0 until iter 300 and 1.5 until
# 600, so the gate sits at 500: 200 iterations of nonzero head incentive.
# KILL conditions, fixed before launch:
#   ABORT  pilot_check hard-fails (positive penalty / NaN / falls blowing up)
#   ABORT  regression: beat_bob or beat_footfall recent mean < 60% of the
#          resumed baseline (first 100 iters) — the fine-tune is eating the
#          dance to feed the head
#   ABORT  head_yaw_power still ~0 after 200 iters at weight >= 1.5
wait_for_iters 500
verdict=2; tries=0
while [ "$verdict" -eq 2 ] && [ "$tries" -lt 6 ]; do
    set +e
    python3 ~/work/duck-mod/scripts/pilot_check.py ~/train_head.log \
        --min-iters 400 --objective head_yaw_power --objective-floor 0.05
    verdict=$?
    set -e
    tries=$((tries + 1))
    [ "$verdict" -eq 2 ] && wait_for_iters $((500 + tries * 100))
done
[ "$verdict" -eq 0 ] || { echo "GATE: pilot_check verdict $verdict — killing"; kill $TRAIN_PID; exit 1; }

python3 - <<'EOF' || { echo "GATE: regression — killing"; kill $TRAIN_PID; exit 1; }
import re, sys
from collections import defaultdict
series = defaultdict(list)
pat = re.compile(r"^\s*Episode_Reward/([\w]+):\s*(-?\d+\.?\d*)\s*$")
import os
for line in open(os.path.expanduser("~/train_head.log"), errors="replace"):
    m = pat.match(line)
    if m:
        series[m.group(1)].append(float(m.group(2)))
ok = True
for name in ("beat_bob", "beat_footfall"):
    s = series.get(name, [])
    if len(s) < 200:
        print(f"{name}: only {len(s)} points"); ok = False; continue
    base = sum(s[:100]) / 100
    recent = sum(s[-100:]) / 100
    print(f"{name}: baseline {base:.3f} recent {recent:.3f}")
    if base > 0 and recent < 0.6 * base:
        ok = False
head = series.get("head_yaw_power", [])
if head:
    print(f"head_yaw_power: recent {sum(head[-50:])/max(len(head[-50:]),1):.3f}")
sys.exit(0 if ok else 1)
EOF

echo "GATE: CONTINUE — letting the fine-tune finish (~28 min total)"
wait $TRAIN_PID
CK=$(ls -t logs/rsl_rl/beat_dance_head/2*/model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -n 1 || true)
[ -n "$CK" ] || CK=$(ls -t wandb/run-*/files/model_*.pt 2>/dev/null | head -n 1 || true)
[ -n "$CK" ] || { echo "no checkpoint found"; exit 1; }
echo "final checkpoint: $CK"
uv run --no-sync scripts/export.py Mjlab-BeatDanceHead-Flat-MicroDuck --checkpoint-file "$CK"
echo "EXPORT_OK output.onnx"

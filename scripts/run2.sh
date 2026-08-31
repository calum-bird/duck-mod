#!/usr/bin/env bash
# Run 2 launch + gate procedure, exactly as designed. Runs on the givemeanode
# node built by setup_node.sh (or restored from snapshot snap-dm756).
#
# Cost structure: the first ~12 minutes ARE the pilot. Killing at the gate
# costs ~$1.30; a full clean run is ~90 min (~$6.00) at 4096 envs.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

cd ~/work/duck-mod && git pull -q origin claude/microduck-training-27jcop
cd ~/work/microduck_rl
# --no-sync everywhere below: `uv run` without it re-syncs the venv to
# microduck_rl's lockfile, which does not know about the dance package.
#
# NEVER pipe a long-lived producer into grep -q / head under pipefail here:
# the consumer exits at first match, the producer takes SIGPIPE, and pipefail
# turns a SUCCESSFUL check into a failed script. That exact bug aborted two
# launches of this script while the tasks were registered and visible in its
# own log. Buffer to a file, then grep the file.
uv sync -q
uv pip install -q --no-deps -e ~/work/duck-mod
uv run --no-sync list-envs > /tmp/envs.txt 2>&1 || true
grep -q Dance /tmp/envs.txt || {
    echo "dance tasks missing after install; retrying verbosely"
    uv pip install --force-reinstall --no-deps -e ~/work/duck-mod
    uv run --no-sync list-envs > /tmp/envs.txt 2>&1 || true
    grep -iE "dance|warn|error|traceback" /tmp/envs.txt || true
    grep -q Dance /tmp/envs.txt || { echo "dance tasks missing"; exit 1; }
}

# CPU-side invariants first: the probe tests fail in seconds if the reward
# stack regressed; the cfg tests re-check every stage-0 weight on real mjlab.
uv pip install -q pytest
uv run --no-sync python -m pytest ~/work/duck-mod/tests/ -q

# WANDB_API_KEY must arrive via the command environment, never this file.
uv run --no-sync train Mjlab-BeatDance-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 3800 \
    > ~/train2.log 2>&1 &
TRAIN_PID=$!
echo "training pid $TRAIN_PID; gate at ~500 iterations (~12 min)"

# ---- The gate -------------------------------------------------------------
# Wait for 500 logged iterations, then judge. KILL conditions, decided before
# launch so the decision is mechanical:
#   ABORT  any Episode_Reward/*_penalty positive, or NaN terminations
#   ABORT  Metrics/peak_height_mean SHRINKING (statue again)
#   ABORT  Episode_Reward/beat_bob_energy flat near zero at iter 500
#           (the escape term is not being collected -> no motion)
#   HOLD   entropy falling fast while energy is flat
# pilot_check encodes all of these.
#
# grep -c prints its count even when it exits nonzero, so `|| echo 0` yields
# "0<newline>0" and [ chokes -- which silently ended this loop on iteration
# zero and had the gate execute a healthy 12-second-old trainer. Count into a
# variable and default it instead.
iters_logged() {
    n=$(grep -c "Iteration time" ~/train2.log 2>/dev/null || true)
    [ -n "$n" ] || n=0
    printf '%s' "$n"
}
wait_for_iters() {
    while [ "$(iters_logged)" -lt "$1" ]; do
        kill -0 $TRAIN_PID 2>/dev/null || { echo "trainer died at $(iters_logged) iterations"; tail -30 ~/train2.log; exit 1; }
        sleep 30
    done
}
wait_for_iters 500
# rc 2 means "too early to judge" -- wait for more iterations, never kill on it.
verdict=2; tries=0
while [ "$verdict" -eq 2 ] && [ "$tries" -lt 6 ]; do
    set +e
    python3 ~/work/duck-mod/scripts/pilot_check.py ~/train2.log \
        --min-iters 400 --objective beat_bob
    verdict=$?
    set -e
    tries=$((tries + 1))
    [ "$verdict" -eq 2 ] && wait_for_iters $((500 + tries * 100))
done
if [ "$verdict" -eq 0 ]; then
    echo "GATE: CONTINUE — letting the run finish (~90 min total)"
    wait $TRAIN_PID
    CK=$(ls -t wandb/run-*/files/model_*.pt 2>/dev/null | head -n 1 || true)
    [ -n "$CK" ] || { echo "no checkpoint found"; exit 1; }
    uv run --no-sync scripts/export.py Mjlab-BeatDance-Flat-MicroDuck --checkpoint-file "$CK"
    echo "exported: output.onnx — eval it on any CPU box:"
    echo "  MICRODUCK_RL=~/work/microduck_rl python ~/work/duck-mod/scripts/eval_dance.py \\"
    echo "      --policy output.onnx --bpm 60 90 120 140 160 --seconds 20"
else
    echo "GATE: verdict above — killing the run"
    kill $TRAIN_PID
    exit 1
fi

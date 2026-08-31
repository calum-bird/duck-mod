#!/usr/bin/env bash
# Bootstrap a givemeanode H100 node for beat-dance training.
#
#   Image: pytorch-2.13-robotics-cuda12.9  (ffmpeg, OpenCV, headless EGL/Vulkan
#   -- the EGL part is what lets us render eval video without a display)
#
# Run once, then snapshot the node: from_snapshot restores the volume AND the
# image it was built under, so every later node skips this entirely. The whole
# point is that no billed training minute is ever spent on `uv sync`.
set -euo pipefail

MICRODUCK_RL_REF="${MICRODUCK_RL_REF:-develop}"
DUCK_MOD_REF="${DUCK_MOD_REF:-claude/microduck-training-27jcop}"
WORK="${WORK:-$HOME/work}"

echo "=== [1/5] uv ==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== [2/5] checkouts ==="
mkdir -p "$WORK" && cd "$WORK"
[ -d microduck_rl ] || git clone --depth 1 -b "$MICRODUCK_RL_REF" \
    https://github.com/pollen-robotics/microduck_rl.git
[ -d duck-mod ] || git clone --depth 1 -b "$DUCK_MOD_REF" \
    https://github.com/calum-bird/duck-mod.git

echo "=== [3/5] deps (the slow part: ~3 GB) ==="
cd "$WORK/duck-mod"
export UV_HTTP_TIMEOUT=600
uv sync

echo "=== [4/5] task registration ==="
uv run list-envs | grep -i dance || {
    echo "!! dance tasks did not register — check the mjlab.tasks entry point"; exit 1; }

echo "=== [5/5] CPU tests + GPU smoke test ==="
uv run --with pytest pytest tests/ -q
uv run train Mjlab-BeatDance-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

cat <<'EOF'

Ready. Next:
  scripts/calibrate_throughput.py     measure steps/s, size the curriculum
  snapshot_node                       so no other node pays for this again

Set MICRODUCK_RL for the viewer/eval scripts:
  export MICRODUCK_RL=$WORK/microduck_rl
EOF

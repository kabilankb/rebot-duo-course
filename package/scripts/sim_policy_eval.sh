#!/usr/bin/env bash
# Evaluate a trained SmolVLA checkpoint in the Isaac Lab bimanual reBot scene:
# a vla-edge server serving the checkpoint, and sim_rollout on unseen layouts.
#
#   scripts/sim_policy_eval.sh CHECKPOINT [EPISODES] [SEED] [-- sim_rollout options...]
#
# CHECKPOINT is a lerobot-train .../checkpoints/<step>/pretrained_model directory.
# Environment overrides: LEROBOT_PYTHON, REBOT_ISAACLAB_PYTHON, REBOT_SIM_PIPELINE, PORT, OUT_DIR.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
positional=()
while (( $# )) && [[ $1 != "--" ]]; do positional+=("$1"); shift; done
[[ ${1:-} == "--" ]] && shift
checkpoint=$(realpath "${positional[0]:?usage: $0 CHECKPOINT [EPISODES] [SEED] [-- sim_rollout options]}")
episodes=${positional[1]:-20}
seed=${positional[2]:-1000}
sim_pipeline=${REBOT_SIM_PIPELINE:-$HOME/rebot-sim-pipeline}
port=${PORT:-8213}
lerobot_python=${LEROBOT_PYTHON:-$HOME/rebot_lerobot/.venv/bin/python}
sim_python=${REBOT_ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_yamlab6/bin/python}
out=${OUT_DIR:-$repo/outputs/eval_$(basename "$(dirname "$checkpoint")")_seed${seed}x${episodes}}
mkdir -p "$out"

if curl -fsS "http://127.0.0.1:$port/healthz" > /dev/null 2>&1; then
    echo "[eval] port $port is already in use" >&2
    exit 1
fi
echo "[eval] server: $checkpoint on :$port (log $out/server.log)"
PYTHONPATH="$repo/src" "$lerobot_python" -m rebot_vla.serve --policy smolvla-bimanual-rebot --backend torch \
    --checkpoint "$checkpoint" --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 180); do
    curl -fsS "http://127.0.0.1:$port/healthz" > /dev/null 2>&1 && break
    kill -0 "$server" 2>/dev/null || { tail -30 "$out/server.log"; exit 1; }
    sleep 1
done
curl -fsS "http://127.0.0.1:$port/act"; echo

echo "[eval] sim client: $episodes episodes from layout seed $seed"
# The demonstrations were recorded with CPU physics; evaluate on the same device.
PYTHONPATH="$repo/src:$sim_pipeline/src" OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$sim_python" -m rebot_vla.sim_rollout --server "127.0.0.1:$port" --device cpu \
    --episodes "$episodes" --seed "$seed" \
    --video "$out/rollout.mp4" --results "$out/results.json" "$@" 2>&1 | tee "$out/rollout.log" \
    | grep --line-buffered -E "^\[(rollout|RESET)\]"

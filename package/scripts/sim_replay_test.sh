#!/usr/bin/env bash
# End-to-end test of the serving path: a vla-edge server replaying one recorded
# episode, and the Isaac Lab bimanual reBot scene driven through POST /act.
# Success means the wire conventions, camera names, chunking and the sim client
# all agree with how the dataset was recorded.
#
#   scripts/sim_replay_test.sh [DATASET_ROOT] [EPISODE] [-- sim_rollout options...]
#
# Environment overrides: VLA_EDGE_PYTHON, REBOT_ISAACLAB_PYTHON, REBOT_SIM_PIPELINE, PORT.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
sim_pipeline=${REBOT_SIM_PIPELINE:-$HOME/rebot-sim-pipeline}
dataset=${1:-$sim_pipeline/datasets/rebot_bimanual_stationery}
episode=${2:-0}
shift $(( $# < 2 ? $# : 2 ))
[[ ${1:-} == "--" ]] && shift
port=${PORT:-8212}
vla_python=${VLA_EDGE_PYTHON:-$HOME/vla-edge/.venv/bin/python}
sim_python=${REBOT_ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_yamlab6/bin/python}
out=${OUT_DIR:-$repo/outputs/replay_ep${episode}}
mkdir -p "$out"

echo "[test] server: replay $dataset episode $episode on :$port (log $out/server.log)"
PYTHONPATH="$repo/src" "$vla_python" -m rebot_vla.serve --checkpoint "$dataset" --episode "$episode" \
    --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 60); do
    curl -fsS "http://127.0.0.1:$port/healthz" > /dev/null 2>&1 && break
    kill -0 "$server" 2>/dev/null || { cat "$out/server.log"; exit 1; }
    sleep 0.5
done
curl -fsS "http://127.0.0.1:$port/act"; echo

echo "[test] sim client (Isaac Lab)"
# Replay is only exact on the device the dataset was recorded on (CPU by default).
PYTHONPATH="$repo/src:$sim_pipeline/src" OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$sim_python" -m rebot_vla.sim_rollout --server "127.0.0.1:$port" --device cpu \
    --video "$out/rollout.mp4" --results "$out/results.json" "$@"

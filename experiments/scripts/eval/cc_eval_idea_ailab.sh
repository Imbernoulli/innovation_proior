#!/usr/bin/env bash
# cc_eval_idea_ailab.sh -- serve one model and run the research-taste task file
# against it inside a single ailab job. Same shape as cc_eval_allinone_ailab.sh,
# whose header explains why one job beats a split serve/client here; this one just
# swaps the FrontierSmith client for scripts/idea_client.py.
#
# gpu_hold is on for the same reason as there. These tasks emit short answers, so
# between waves the card can sit idle long enough for the sweep to fire, and a
# 1-token ping does NOT count as utilisation -- see the comment block in
# cc_eval_allinone_ailab.sh for the job that died proving it.
#
#   sbatch --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G \
#     --time=06:00:00 --job-name=idea-<TAG> \
#     --output=$D/logs/%x-%j.out --error=$D/logs/%x-%j.out \
#     --export=ALL,MODEL=<dir>,TAG=<tag> cc_eval_idea_ailab.sh
set -uo pipefail

D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
PROJECT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$PROJECT_ROOT"

: "${MODEL:?set MODEL}" "${TAG:?set TAG}"
export VLLM_VENV="${VLLM_VENV:-$D/envs/vllm023}"
export PATH="$D/envs/client/bin:$PATH"
export HF_HOME="${HF_HOME:-$D/.hf}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$D/.cache/vllm}"
export TP="${TP:-1}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FS_VLLM_PENALTY_FASTPATH=1
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"

TASKS="${TASKS_FILE:-$D/ideabench/tasks.jsonl}"
OUT="${OUT_DIR:-$D/outputs/cc_idea_$TAG}"
[ -f "$TASKS" ] || { echo "no task file $TASKS (run scripts/idea_prep.py first)" >&2; exit 2; }

VLLM_PORT="${VLLM_PORT:-$(python3 -c '
import socket
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
export VLLM_PORT

cleanup() {
  [ -n "${KEEPALIVE_PID:-}" ] && kill "$KEEPALIVE_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[idea] TAG=$TAG model=$MODEL port=$VLLM_PORT out=$OUT"
HOST=127.0.0.1 PORT="$VLLM_PORT" MODEL_PATH="$MODEL" SERVED_MODEL_NAME="$TAG" \
  bash scripts/start_vllm_server.sh &
VLLM_PID=$!

ready=0
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[idea] vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "[idea] vLLM never became ready" >&2; exit 1; }
echo "[idea] vLLM ready after $SECONDS s"

GPU_HOLD_PY="${GPU_HOLD_PY:-$HOME/gpu_hold/run.py}"
KEEPALIVE_PID=""
if [ -f "$GPU_HOLD_PY" ]; then
  HOLD_SIZE=4096 HOLD_SLEEP=20 HOLD_BURST=2 \
    "$D/envs/vllm023/bin/python" "$GPU_HOLD_PY" >/dev/null 2>&1 &
  KEEPALIVE_PID=$!
  echo "[idea] gpu_hold keep-alive started (pid $KEEPALIVE_PID)"
else
  echo "[idea] WARNING: $GPU_HOLD_PY missing; the idle sweep may kill this job" >&2
fi

"$D/envs/client/bin/python" "$D/scripts/idea_client.py" \
  --tasks "$TASKS" --out "$OUT" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" --model "$TAG" \
  --n-samples "${N_SAMPLES:-5}" --concurrency "${CONCURRENCY:-48}"
rc=$?
echo "[idea] client exited rc=$rc after $SECONDS s"
exit $rc

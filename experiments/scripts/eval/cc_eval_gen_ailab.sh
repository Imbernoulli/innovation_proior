#!/usr/bin/env bash
# cc_eval_gen_ailab.sh -- serve one model and run the GENERATION-side research-taste
# tasks against it (scripts/gen_client.py + ideabench/gentasks.jsonl).
#
# Same shape as cc_eval_idea_ailab.sh; see that file's header for why serve+client live
# in one job and why gpu_hold is mandatory. The only differences here are the client and
# the token budget: MAX_TOKENS defaults to 32768, the length these models were trained
# at, because the 8192 default silently turned the multiple-choice tasks into a race to
# finish thinking (base9b_v2c AAAR 0.4617 at 8192 -> 0.6200 at 24576).
#
#   sbatch --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G \
#     --time=10:00:00 --job-name=gen-<TAG> \
#     --output=$D/logs/%x-%j.out --error=$D/logs/%x-%j.out \
#     --export=ALL,MODEL=<dir>,TAG=<tag> cc_eval_gen_ailab.sh
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
# prompts here run to ~2.5k tokens (title + 2.2k-char abstract), so 45056 leaves the
# full 32768 completion budget reachable on every item rather than most of them.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-45056}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"

TASKS="${TASKS_FILE:-$D/ideabench/gentasks.jsonl}"
OUT="${OUT_DIR:-$D/outputs/cc_gen_$TAG}"
[ -f "$TASKS" ] || { echo "no task file $TASKS (run scripts/gen_prep.py first)" >&2; exit 2; }

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

echo "[gen] TAG=$TAG model=$MODEL port=$VLLM_PORT out=$OUT"
HOST=127.0.0.1 PORT="$VLLM_PORT" MODEL_PATH="$MODEL" SERVED_MODEL_NAME="$TAG" \
  bash scripts/start_vllm_server.sh &
VLLM_PID=$!

ready=0
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[gen] vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "[gen] vLLM never became ready" >&2; exit 1; }
echo "[gen] vLLM ready after $SECONDS s"

GPU_HOLD_PY="${GPU_HOLD_PY:-$HOME/gpu_hold/run.py}"
KEEPALIVE_PID=""
if [ -f "$GPU_HOLD_PY" ]; then
  HOLD_SIZE=4096 HOLD_SLEEP=20 HOLD_BURST=2 \
    "$D/envs/vllm023/bin/python" "$GPU_HOLD_PY" >/dev/null 2>&1 &
  KEEPALIVE_PID=$!
  echo "[gen] gpu_hold keep-alive started (pid $KEEPALIVE_PID)"
else
  echo "[gen] WARNING: $GPU_HOLD_PY missing; the idle sweep may kill this job" >&2
fi

"$D/envs/client/bin/python" "$D/scripts/gen_client.py" \
  --tasks "$TASKS" --out "$OUT" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" --model "$TAG" \
  --max-tokens "${MAX_TOKENS:-32768}" \
  ${ONLY_TASK:+--only-task "$ONLY_TASK"} \
  --n-samples "${N_SAMPLES:-4}" --concurrency "${CONCURRENCY:-48}"
rc=$?
echo "[gen] client exited rc=$rc after $SECONDS s"
exit $rc

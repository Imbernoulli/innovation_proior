#!/usr/bin/env bash
# cc_judgepw_ailab.sh -- host the un-finetuned Qwen and score candidates ONE AT A TIME.
#
# Same job shape as cc_judge_ailab.sh; the difference is judge_pointwise.py instead of
# judge_pairwise.py. Use this where the pairwise judge demonstrably failed its own
# position-bias check: on review_weakness it picked the second-shown candidate in 87%
# of calls in BOTH orders, so only 23.1% of pairs got a consistent verdict. Scoring one
# candidate at a time removes that failure mode by construction.
#
# PAIRS is OURTAG:CONTROLTAG, same as the pairwise job. TASKS_ONLY selects which task
# to score (default review_weakness) -- pointwise is the fallback, not the default, so
# do not sweep it over tasks where pairwise already agrees with itself.
#
#   sbatch --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G \
#     --time=10:00:00 --job-name=judgepw \
#     --output=$D/logs/%x-%j.out --error=$D/logs/%x-%j.out \
#     --export=ALL,PAIRS="ft01mix_a10:base9b_v2c" cc_judgepw_ailab.sh
set -uo pipefail

D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
PROJECT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$PROJECT_ROOT"

: "${PAIRS:?set PAIRS='our:control our2:control2'}"
JUDGE_MODEL="${JUDGE_MODEL:-$PROJECT_ROOT/models/Qwen3.5-9B-bf16}"
JUDGE_TAG="${JUDGE_TAG:-judge9b}"
TASK="${TASKS_ONLY:-review_weakness}"

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
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"

TASKS="${TASKS_FILE:-$D/ideabench/gentasks.jsonl}"
[ -f "$TASKS" ] || { echo "no task file $TASKS" >&2; exit 2; }

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

echo "[pw] judge=$JUDGE_MODEL task=$TASK port=$VLLM_PORT"
echo "[pw] pairs: $PAIRS"
HOST=127.0.0.1 PORT="$VLLM_PORT" MODEL_PATH="$JUDGE_MODEL" SERVED_MODEL_NAME="$JUDGE_TAG" \
  bash scripts/start_vllm_server.sh &
VLLM_PID=$!

ready=0
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[pw] vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "[pw] vLLM never became ready" >&2; exit 1; }
echo "[pw] vLLM ready after $SECONDS s"

GPU_HOLD_PY="${GPU_HOLD_PY:-$HOME/gpu_hold/run.py}"
KEEPALIVE_PID=""
if [ -f "$GPU_HOLD_PY" ]; then
  HOLD_SIZE=4096 HOLD_SLEEP=20 HOLD_BURST=2 \
    "$D/envs/vllm023/bin/python" "$GPU_HOLD_PY" >/dev/null 2>&1 &
  KEEPALIVE_PID=$!
fi

rc=0
for pair in $PAIRS; do
  ours="${pair%%:*}"; ctrl="${pair##*:}"
  ad="$D/outputs/cc_gen_$ours"; bd="$D/outputs/cc_gen_$ctrl"
  if [ ! -f "$ad/samples.jsonl" ] || [ ! -f "$bd/samples.jsonl" ]; then
    echo "[pw] SKIP $ours vs $ctrl -- missing generation output" >&2
    rc=1; continue
  fi
  echo "[pw] === $ours vs $ctrl ($TASK) ==="
  "$D/envs/client/bin/python" "$D/scripts/judge_pointwise.py" \
    --a-dir "$ad" --b-dir "$bd" --a-tag "$ours" --b-tag "$ctrl" --task "$TASK" \
    --tasks "$TASKS" --out "$D/outputs/cc_judgepw_${TASK}${OUT_SUFFIX:-}_${ours}__vs__${ctrl}" \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" --model "$JUDGE_TAG" \
    --max-tokens "${MAX_TOKENS:-16384}" \
    --n-pairs "${N_PAIRS:-4}" --concurrency "${CONCURRENCY:-32}" || rc=$?
done

echo "[pw] done rc=$rc after $SECONDS s"
exit $rc

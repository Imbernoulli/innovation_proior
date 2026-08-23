#!/usr/bin/env bash
# Qwen3.5-9B base evaluation through an already-running vLLM server.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "ERROR: .venv not found. Run setup-env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1

export ALE_BENCH_DATA="${ALE_BENCH_DATA:-$PROJECT_ROOT/data/alebench/local_data}"
export ALE_BENCH_CACHE="${ALE_BENCH_CACHE:-$PROJECT_ROOT/.cache/ale-bench}"
export ALE_BENCH_TOOL_CACHE="${ALE_BENCH_TOOL_CACHE:-$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds}"
export ALE_BENCH_REQUIRE_TOOL_CACHE="${ALE_BENCH_REQUIRE_TOOL_CACHE:-1}"
export ALE_BENCH_CONTAINER_BACKEND="${ALE_BENCH_CONTAINER_BACKEND:-apptainer}"
export ALE_BENCH_APPTAINER_DIR="${ALE_BENCH_APPTAINER_DIR:-$PROJECT_ROOT/.cache/apptainer/alebench}"
export ALEBENCH_JUDGE_VERSION="${ALEBENCH_JUDGE_VERSION:-202301}"
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

FRONTIERCS_VAL_DATA="${FRONTIERCS_VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}"
# ALE is ALWAYS the 40-problem set (user directive 2026-08-22, standing).
# The shipped val.parquet holds only 10 of ALE-Bench's 40 problems; its
# difference-SEM (~96 points) made every cross-arm ALE claim unusable, so the
# 10-problem set is retired. ALEBENCH_LITE=false is what makes the harness read
# the full set rather than its "lite" 10.
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/full40.parquet}"
export ALEBENCH_LITE="${ALEBENCH_LITE:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/base_eval_qwen35_9b_vllm}"
N_SAMPLES="${N_SAMPLES:-5}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
SOURCE="${SOURCE:-both}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
CONCURRENCY="${CONCURRENCY:-16}"

for path in "$FRONTIERCS_VAL_DATA" "$ALEBENCH_VAL_DATA" "$ALE_BENCH_DATA" "$ALE_BENCH_APPTAINER_DIR" "$ALE_BENCH_TOOL_CACHE"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

args=(
  --frontiercs-data "$FRONTIERCS_VAL_DATA"
  --alebench-data "$ALEBENCH_VAL_DATA"
  --source "$SOURCE"
  --output-dir "$OUTPUT_DIR"
  --num-shards "$NUM_SHARDS"
  --shard-idx "$SHARD_IDX"
  --n-samples "$N_SAMPLES"
  --base-url "$VLLM_BASE_URL"
  --model "$SERVED_MODEL_NAME"
  --max-tokens "$MAX_TOKENS"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-1.0}"
  --concurrency "$CONCURRENCY"
  --timeout "${REQUEST_TIMEOUT:-1800}"
  --judge-url "${FRONTIERCS_JUDGE_URL:-http://127.0.0.1:${PORT:-8082}}"
  --frontiercs-prompt-source "${FRONTIERCS_PROMPT_SOURCE:-official}"
  --frontiercs-score-backend "${FRONTIERCS_SCORE_BACKEND:-official}"
  --frontiercs-iterative-rounds "${FRONTIERCS_ITERATIVE_ROUNDS:-1}"
  --seed "${SEED:-42}"
  --max-errors "${MAX_ERRORS:-0}"
)

if [ -n "${SAMPLES_JSONL:-}" ]; then
  args+=(--samples-jsonl "$SAMPLES_JSONL")
fi
if [ -n "${SUMMARY_JSON:-}" ]; then
  args+=(--summary-json "$SUMMARY_JSON")
fi
if [ "${RESUME:-1}" = "1" ]; then
  args+=(--resume)
fi
# Decouple generation from scoring so a slow judge never starves the vLLM engine
# (the ~45% wall-clock stall fix). Default ON; set EVAL_DECOUPLE=0 to fall back to
# the old coupled loop. EVAL_SCORE_CONCURRENCY sizes the (CPU-bound) scorer pool.
if [ "${EVAL_DECOUPLE:-1}" = "1" ]; then
  args+=(--decouple-scoring)
fi
if [ -n "${FRONTIERCS_LIMIT:-}" ]; then
  args+=(--limit-frontiercs "$FRONTIERCS_LIMIT")
fi
if [ -n "${ALEBENCH_LIMIT:-}" ]; then
  args+=(--limit-alebench "$ALEBENCH_LIMIT")
fi
if [ -n "${TOP_K:-}" ]; then
  args+=(--top-k "$TOP_K")
fi
if [ -n "${MIN_P:-}" ]; then
  args+=(--min-p "$MIN_P")
fi
if [ -n "${PRESENCE_PENALTY:-}" ]; then
  args+=(--presence-penalty "$PRESENCE_PENALTY")
fi
if [ -n "${FREQUENCY_PENALTY:-}" ]; then
  args+=(--frequency-penalty "$FREQUENCY_PENALTY")
fi
if [ -n "${REPETITION_PENALTY:-}" ]; then
  args+=(--repetition-penalty "$REPETITION_PENALTY")
fi
if [ "${ENABLE_THINKING:-}" = "0" ]; then
  args+=(--no-enable-thinking)
elif [ "${ENABLE_THINKING:-}" = "1" ]; then
  args+=(--enable-thinking)
fi
if [ "${SAVE_TEXT:-1}" = "0" ]; then
  args+=(--no-save-text)
elif [ "${SAVE_TEXT:-1}" = "1" ]; then
  args+=(--save-text)
fi
if [ -n "${TEXT_PREVIEW_CHARS:-}" ]; then
  args+=(--text-preview-chars "$TEXT_PREVIEW_CHARS")
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  args+=(--dry-run)
fi

python scripts/eval_qwen35_base_vllm_request.py "${args[@]}" "$@"

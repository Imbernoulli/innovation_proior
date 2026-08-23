#!/usr/bin/env bash
# Offline Qwen3.5-9B base evaluation with Transformers generation.

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

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
FRONTIERCS_VAL_DATA="${FRONTIERCS_VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}"
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/base_eval_qwen35_9b_hf}"
N_SAMPLES="${N_SAMPLES:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
FRONTIERCS_LIMIT="${FRONTIERCS_LIMIT:-}"
ALEBENCH_LIMIT="${ALEBENCH_LIMIT:-}"
SOURCE="${SOURCE:-both}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"

for path in "$MODEL_PATH" "$FRONTIERCS_VAL_DATA" "$ALEBENCH_VAL_DATA" "$ALE_BENCH_DATA" "$ALE_BENCH_APPTAINER_DIR" "$ALE_BENCH_TOOL_CACHE"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

args=(
  --model "$MODEL_PATH"
  --frontiercs-data "$FRONTIERCS_VAL_DATA"
  --alebench-data "$ALEBENCH_VAL_DATA"
  --source "$SOURCE"
  --output-dir "$OUTPUT_DIR"
  --num-shards "$NUM_SHARDS"
  --shard-idx "$SHARD_IDX"
  --n-samples "$N_SAMPLES"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-1.0}"
  --top-k "${TOP_K:-0}"
  --dtype "$DTYPE"
  --device-map "$DEVICE_MAP"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --judge-url "${FRONTIERCS_JUDGE_URL:-http://127.0.0.1:${PORT:-8082}}"
  --seed "${SEED:-42}"
)

if [ "${RESUME:-1}" = "1" ]; then
  args+=(--resume)
fi
if [ -n "$FRONTIERCS_LIMIT" ]; then
  args+=(--limit-frontiercs "$FRONTIERCS_LIMIT")
fi
if [ -n "$ALEBENCH_LIMIT" ]; then
  args+=(--limit-alebench "$ALEBENCH_LIMIT")
fi
if [ "${ENABLE_THINKING:-}" = "0" ]; then
  args+=(--no-enable-thinking)
elif [ "${ENABLE_THINKING:-}" = "1" ]; then
  args+=(--enable-thinking)
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  args+=(--dry-run)
fi

python scripts/eval_base_model_qwen35_hf.py "${args[@]}" "$@"

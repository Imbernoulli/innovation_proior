#!/bin/bash
# Start vLLM OpenAI-compatible server for FrontierSmith evaluation.
#
# Usage:
#   MODEL_PATH=models/Qwen3.5-9B bash scripts/start_vllm_server.sh
#   MODEL_PATH=models/qwen35_9b_grpo_step105 PORT=8000 bash scripts/start_vllm_server.sh
#
# Uses CUDA_VISIBLE_DEVICES from Slurm or the caller. Stop with Ctrl+C.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VLLM_VENV="${VLLM_VENV:-$PROJECT_ROOT/.venv-vllm023}"
if [ -f "$VLLM_VENV/bin/activate" ]; then
    source "$VLLM_VENV/bin/activate"
fi

MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen35-9b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-26624}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
DTYPE=${DTYPE:-bfloat16}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-1}
# Prefix caching: OFF by default to preserve byte-identical behaviour for the
# existing single-job evals, ON for the shared pool backend (see below).
# Value here is large: every problem is sent n_samples times (5, and 32 for the
# Research top-up) with an identical prompt, and a pool backend additionally sees
# the same problems from several client jobs. MLS-Bench is multi-turn and shares a
# growing prefix across turns. Measured caveat, so nobody over-expects it: for
# short single-turn prompts (~900 tokens in training rollouts) prefill is only a
# few percent of wall-clock -- this speeds up prefill, it does not fix decode.
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
# The vLLM penalty fast path (presence_penalty=1.5 runs the penalty kernel every
# decode step; the stock path rebuilds a [batch, max_len] tensor each time).
# Verified active via the "[fs] vLLM penalty fast path ACTIVE" log line.
export FS_VLLM_PENALTY_FASTPATH="${FS_VLLM_PENALTY_FASTPATH:-1}"

if [ -z "${GDN_PREFILL_BACKEND+x}" ]; then
    GDN_PREFILL_BACKEND="$(
        MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os
from pathlib import Path

try:
    model_type = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text()).get("model_type")
except Exception:
    model_type = None

print("triton" if model_type in {"qwen3_5", "qwen3_5_moe"} else "")
PY
    )"
fi

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$PROJECT_ROOT/.cache/vllm}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
    "$MODEL_PATH"
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --trust-remote-code
)

# --served-model-name takes nargs='+': a request matches ANY registered name.
# Expand SERVED_MODEL_NAME UNQUOTED so a space-separated value (e.g.
# "TAG vllm/TAG") registers MULTIPLE aliases. A single-token value (the common
# case for FrontierCS/ALE/Theta evals, e.g. "qwen35-9b") word-splits to one arg,
# so those callers are unaffected. This lets both MLS-Bench-dev (sends bare TAG,
# prefix stripped) and non-dev MLS-Bench (sends vllm/TAG) hit a served name.
# shellcheck disable=SC2206
args+=(--served-model-name $SERVED_MODEL_NAME)

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    args+=(--enable-prefix-caching)
fi

if [ -n "$GDN_PREFILL_BACKEND" ]; then
    args+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
fi

if [ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
    args+=(--disable-custom-all-reduce)
fi

echo "Starting vLLM server: model=$MODEL_PATH served=$SERVED_MODEL_NAME host=$HOST port=$PORT TP=$TP max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
exec vllm serve "${args[@]}" "$@"

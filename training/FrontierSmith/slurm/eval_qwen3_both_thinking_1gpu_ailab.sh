#!/usr/bin/env bash
# Full FrontierCS + ALE-Bench eval for one Qwen3-family model on one H200.
#
# This is the non-smoke path used for Qwen3-8B, Qwen3-8B-Base, and their
# linear merges. It uses the official FrontierCS prompt/scorer and Qwen
# thinking-general sampling defaults.
#
# Submit example:
#   MODEL_PATH=models/Qwen3-8B MODEL_TAG=qwen3_8b \
#     sbatch slurm/eval_qwen3_both_thinking_1gpu_ailab.sh

#SBATCH --job-name=fs-qwen3-eval
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-8B}"
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_PATH" | tr -cs 'A-Za-z0-9._-' '_')}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_TAG}"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TMPDIR="/tmp"

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

if [ -z "${PORT_OFFSET+x}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  PORT_OFFSET="$((SLURM_JOB_ID % 10000))"
else
  PORT_OFFSET="${PORT_OFFSET:-0}"
fi
export JUDGE_PORT="${JUDGE_PORT:-$((8082 + PORT_OFFSET))}"
export VLLM_PORT="${VLLM_PORT:-$((8000 + PORT_OFFSET))}"
export GJ_PORT="${GJ_PORT:-$((5050 + PORT_OFFSET))}"
export PORT="$JUDGE_PORT"
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-qwen3-eval-${SLURM_JOB_ID:-manual}}"

if [ -z "${MAX_MODEL_LEN:-}" ]; then
  MAX_MODEL_LEN="$(
    MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os
from pathlib import Path

default = 49152
try:
    cfg = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
    max_pos = cfg.get("max_position_embeddings")
    if isinstance(max_pos, int) and max_pos > 0:
        print(min(default, max_pos))
    else:
        print(default)
except Exception:
    print(default)
PY
  )"
fi
export MAX_MODEL_LEN

MAX_TOKENS="${MAX_TOKENS:-32768}"
if [ -z "${REQUEST_MAX_TOKENS:-}" ]; then
  # Official FrontierCS/ALE prompts reach ~8.2K Qwen tokens. Keep a small
  # margin so the longest prompt does not exceed model context at request time.
  reserve="${MAX_PROMPT_RESERVE:-8704}"
  safe_max_tokens=$((MAX_MODEL_LEN - reserve))
  if [ "$safe_max_tokens" -lt 1024 ]; then
    safe_max_tokens=1024
  fi
  REQUEST_MAX_TOKENS="$MAX_TOKENS"
  if [ "$safe_max_tokens" -lt "$REQUEST_MAX_TOKENS" ]; then
    REQUEST_MAX_TOKENS="$safe_max_tokens"
  fi
fi
export REQUEST_MAX_TOKENS

export MODEL_PATH
export SERVED_MODEL_NAME
export TP="${TP:-1}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

HOST=127.0.0.1 PORT="$VLLM_PORT" scripts/start_vllm_server.sh &
VLLM_PID="$!"

cleanup() {
  kill "$VLLM_PID" "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$VLLM_PID" "$JUDGE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "$JUDGE_PID" >/dev/null 2>&1; then
    echo "FrontierCS judge launcher exited early" >&2
    exit 1
  fi
done
curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health"
echo

vllm_models_json=""
for _ in $(seq 1 720); do
  if vllm_models_json="$(curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null)" \
    && printf '%s\n' "$vllm_models_json" | grep -F "\"id\":\"$SERVED_MODEL_NAME\"" >/dev/null; then
    break
  fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM server exited early" >&2
    exit 1
  fi
done
if ! printf '%s\n' "$vllm_models_json" | grep -F "\"id\":\"$SERVED_MODEL_NAME\"" >/dev/null; then
  echo "Timed out waiting for vLLM model '$SERVED_MODEL_NAME' on port $VLLM_PORT" >&2
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" || true
  exit 1
fi
printf '%s\n' "$vllm_models_json"
echo

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_${MODEL_TAG}_thinking_general_both_vllm}" \
SOURCE="${SOURCE:-both}" \
N_SAMPLES="${N_SAMPLES:-5}" \
MAX_TOKENS="$REQUEST_MAX_TOKENS" \
CONCURRENCY="${CONCURRENCY:-64}" \
ENABLE_THINKING="${ENABLE_THINKING:-1}" \
TEMPERATURE="${TEMPERATURE:-1.0}" \
TOP_P="${TOP_P:-0.95}" \
TOP_K="${TOP_K:-20}" \
MIN_P="${MIN_P:-0.0}" \
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}" \
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}" \
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-2400}" \
FRONTIERCS_PROMPT_SOURCE="${FRONTIERCS_PROMPT_SOURCE:-official}" \
FRONTIERCS_SCORE_BACKEND="${FRONTIERCS_SCORE_BACKEND:-official}" \
SAVE_TEXT="${SAVE_TEXT:-0}" \
TEXT_PREVIEW_CHARS="${TEXT_PREVIEW_CHARS:-4000}" \
VLLM_PORT="$VLLM_PORT" \
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
PORT="$JUDGE_PORT" \
bash scripts/eval_base_model_qwen35_9b_vllm_request.sh "$@"

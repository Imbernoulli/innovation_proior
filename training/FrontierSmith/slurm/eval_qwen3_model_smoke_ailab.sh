#!/usr/bin/env bash
#SBATCH --job-name=fs-qwen3-eval
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_CACHE_DIR="${PWD}/.cache/vllm"
export HF_HOME="${PWD}/.cache/huggingface"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-${PWD}/models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-${PWD}/data/frontiercs/train_synthetic.parquet}"
EVAL_NAME="${EVAL_NAME:-$(basename "${MODEL_PATH}")}"
VLLM_PORT="${VLLM_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8082}"
GJ_PORT="${GJ_PORT:-5050}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2304}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CONCURRENCY="${CONCURRENCY:-2}"
N_SAMPLES="${N_SAMPLES:-1}"

export PORT="${JUDGE_PORT}"
export GJ_PORT
export GJ_PARALLELISM="${GJ_PARALLELISM:-4}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.cache/frontiercs-judge-eval-${SLURM_JOB_ID:-manual}}"


scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "${VLLM_PID:-}" >/dev/null 2>&1 || true
  kill "${JUDGE_PID}" >/dev/null 2>&1 || true
  wait "${VLLM_PID:-}" >/dev/null 2>&1 || true
  wait "${JUDGE_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health"
echo

python scripts/vllm_serve_compat.py serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2304 &
VLLM_PID="$!"

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "vLLM exited early" >&2
    exit 1
  fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null

mkdir -p outputs/eval
python scripts/eval_frontiercs_via_vllm.py \
  --data "${DATA_PATH}" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --judge-url "http://127.0.0.1:${JUDGE_PORT}" \
  --n-samples "${N_SAMPLES}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --output-csv "outputs/eval/${EVAL_NAME}_smoke.csv" \
  --print-csv-row

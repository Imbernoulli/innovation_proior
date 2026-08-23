#!/usr/bin/env bash
#SBATCH --job-name=fs-q35-train
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
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

export VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-vllm023}"
source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_CACHE_DIR="$PROJECT_ROOT/.cache/vllm"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${SLURM_JOB_ID}}"
export TMPDIR="/tmp"

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-4}"

export PORT="${PORT:-8082}"
export GJ_PORT="${GJ_PORT:-5050}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-train-${SLURM_JOB_ID}}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$JUDGE_PID" >/dev/null 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "$JUDGE_PID" >/dev/null 2>&1; then
    echo "FrontierCS judge launcher exited early" >&2
    exit 1
  fi
done
curl -fsS "http://127.0.0.1:${PORT}/health"
echo

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}" \
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}" \
VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}" \
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}" \
NGPU="${NGPU:-4}" \
TP="${TP:-1}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16000}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-26624}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" \
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}" \
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}" \
bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh "$@"

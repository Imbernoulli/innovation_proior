#!/usr/bin/env bash
#SBATCH --job-name=fs-q35-base-eval
#SBATCH --partition=ailab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
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
source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_CACHE_DIR="$PROJECT_ROOT/.cache/vllm"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

export PORT="${PORT:-8082}"
export GJ_PORT="${GJ_PORT:-5050}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-8}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-base-eval-${SLURM_JOB_ID}}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$JUDGE_PID" >/dev/null 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
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

NGPU="${NGPU:-2}" \
TP="${TP:-2}" \
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}" \
FRONTIERCS_VAL_DATA="${FRONTIERCS_VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}" \
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}" \
RAY_TMPDIR="$RAY_TMPDIR" \
bash scripts/eval_base_model_qwen35_9b.sh "$@"

#!/usr/bin/env bash
#SBATCH --job-name=fs-q35-hf-array
#SBATCH --partition=ailab
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

mkdir -p logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-8}}"
BASE_PORT="${BASE_PORT:-8082}"
BASE_GJ_PORT="${BASE_GJ_PORT:-5050}"
export PORT="${PORT:-$((BASE_PORT + SHARD_IDX))}"
export GJ_PORT="${GJ_PORT:-$((BASE_GJ_PORT + SHARD_IDX))}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-8}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-hf-array-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SHARD_IDX}}"
export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$JUDGE_PID" >/dev/null 2>&1 || true
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

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}" \
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/base_eval_qwen35_9b_hf/shards/shard_${SHARD_IDX}}" \
N_SAMPLES="${N_SAMPLES:-5}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}" \
NUM_SHARDS="$NUM_SHARDS" \
SHARD_IDX="$SHARD_IDX" \
ENABLE_THINKING="${ENABLE_THINKING:-0}" \
RESUME="${RESUME:-1}" \
bash scripts/eval_base_model_qwen35_9b_hf.sh "$@"

#!/usr/bin/env bash
#SBATCH --job-name=fs-q35-hf-smoke
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
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

export PORT="${PORT:-8082}"
export GJ_PORT="${GJ_PORT:-5050}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-4}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-hf-smoke-${SLURM_JOB_ID}}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/base_eval_qwen35_9b_hf_smoke/${SLURM_JOB_ID}}" \
N_SAMPLES="${N_SAMPLES:-1}" \
FRONTIERCS_LIMIT="${FRONTIERCS_LIMIT:-1}" \
ALEBENCH_LIMIT="${ALEBENCH_LIMIT:-1}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}" \
RESUME="${RESUME:-0}" \
bash scripts/eval_base_model_qwen35_9b_hf.sh "$@"

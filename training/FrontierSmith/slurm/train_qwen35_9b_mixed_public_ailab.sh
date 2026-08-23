#!/usr/bin/env bash
# GRPO training for Qwen3.5-9B on the public/released mix:
#   FrontierCS 172 + FrontierSmith synthetic 10 + ALE-Bench full public 40.
#
# Submit with:
#   sbatch slurm/train_qwen35_9b_mixed_public_ailab.sh

#SBATCH --job-name=fs-q35-mixtrain
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:30:00
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
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${SLURM_JOB_ID}}"
export TMPDIR="/tmp"

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_REQUIRE_TOOL_CACHE="${ALE_BENCH_REQUIRE_TOOL_CACHE:-1}"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-4}"

python scripts/prepare_alebench_tool_cache.py --no-lite --check-only >/dev/null

if [ -z "${PORT_OFFSET+x}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  PORT_OFFSET="$((SLURM_JOB_ID % 10000))"
else
  PORT_OFFSET="${PORT_OFFSET:-0}"
fi
export PORT="${PORT:-$((8082 + PORT_OFFSET))}"
export GJ_PORT="${GJ_PORT:-$((5050 + PORT_OFFSET))}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-mixtrain-${SLURM_JOB_ID}}"

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

export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rollout_data_qwen35_9b_mixed_public}"
export PROJECT_NAME="${PROJECT_NAME:-verl_frontiercs_qwen35_9b_mixed_public}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_mixed_public}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-25}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export VAL_N="${VAL_N:-5}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}" \
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet}" \
VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}" \
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}" \
NGPU="${NGPU:-4}" \
TP="${TP:-1}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}" \
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}" \
FRESH_START="${FRESH_START:-0}" \
bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh "$@"

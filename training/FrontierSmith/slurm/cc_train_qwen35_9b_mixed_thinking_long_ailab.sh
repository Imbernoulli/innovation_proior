#!/usr/bin/env bash
# THINKING-ENABLED, long GRPO training for Qwen3.5-9B on the public mix
# (FrontierCS 172 + FrontierSmith synthetic 10 + ALE-Bench full public 40).
#
# Why this script exists (vs slurm/train_qwen35_9b_mixed_public_ailab.sh):
#   The earlier mixed run scored 0.0 after training. Root cause (verified on its
#   rollout dumps): Qwen3.5 chat template defaults to THINKING ON, so rollouts
#   generate long <think> traces — but MAX_RESPONSE_LENGTH=16000 truncated 71% of
#   them before </think> closed, so 57% of rollouts produced NO extractable code
#   => near-zero reward => GRPO degraded the model.
#
#   Fix: give thinking room (response 30000, max_model_len 40960), keep thinking
#   on explicitly, train longer with frequent checkpoints. Eval is done WITH
#   thinking via slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh.
#
# Submit:
#   sbatch slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh

#SBATCH --job-name=cc-q35-mixthink-long
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=23:59:00
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
export ALE_BENCH_REQUIRE_TOOL_CACHE=1
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-4}"

# Judge ports (per-node; keep distinct from the default wrapper to be safe).
export PORT="${PORT:-8092}"
export GJ_PORT="${GJ_PORT:-5060}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-ccthink-${SLURM_JOB_ID}}"

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

# Own checkpoint/rollout/project namespace so it never collides with Codex's runs
# (their FRESH_START backs up/wipes the other checkpoint dir, not this one).
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/cc_verl_qwen35_9b_thinking/qwen35_9b_grpo_mixed_thinking}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/cc_rollout_qwen35_9b_mixed_thinking}"
export PROJECT_NAME="${PROJECT_NAME:-cc_verl_qwen35_9b_thinking}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_mixed_thinking}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-40}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
# Skip in-loop validation (it would be non-thinking and waste rollout time);
# we evaluate saved checkpoints WITH thinking separately.
export TEST_FREQ="${TEST_FREQ:-100000}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VAL_N="${VAL_N:-1}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}" \
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet}" \
VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}" \
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}" \
NGPU="${NGPU:-4}" \
TP="${TP:-1}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-60}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-30000}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-40960}" \
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}" \
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}" \
FRESH_START="${FRESH_START:-1}" \
bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh \
  ++data.apply_chat_template_kwargs.enable_thinking=True \
  "$@"

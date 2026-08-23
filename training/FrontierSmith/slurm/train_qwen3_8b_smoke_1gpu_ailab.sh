#!/usr/bin/env bash
#SBATCH --job-name=fs-qwen3-smoke1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=360G
#SBATCH --time=00:30:00
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
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_CACHE_DIR="${PWD}/.cache/vllm"
export HF_HOME="${PWD}/.cache/huggingface"
export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1

export PORT="${PORT:-8082}"
export GJ_PORT="${GJ_PORT:-5050}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-4}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.cache/frontiercs-judge-train-smoke-${SLURM_JOB_ID}}"

MODEL_PATH="${MODEL_PATH:-${PWD}/models/Qwen3-8B}"
TRAIN_DATA="${TRAIN_DATA:-${PWD}/data/frontiercs/train_synthetic.parquet}"
VAL_DATA="${VAL_DATA:-${PWD}/data/frontiercs/train_synthetic.parquet}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-$(basename "${MODEL_PATH}" | tr -cs 'A-Za-z0-9._-' '_')}"
PROJECT_NAME="${PROJECT_NAME:-frontiersmith_qwen3_8b_smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_RUN_NAME}_grpo_smoke_1gpu}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PWD}/checkpoints/${MODEL_RUN_NAME}_smoke_1gpu}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PWD}/outputs/rollout_data_${MODEL_RUN_NAME}_smoke_1gpu}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
ROLLOUT_N="${ROLLOUT_N:-2}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2304}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
RESUME_MODE="${RESUME_MODE:-disable}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "${JUDGE_PID}" >/dev/null 2>&1 || true
  wait "${JUDGE_PID}" >/dev/null 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "${JUDGE_PID}" >/dev/null 2>&1; then
    echo "judge launcher exited early" >&2
    exit 1
  fi
done

curl -fsS "http://127.0.0.1:${PORT}/health"
echo

MODEL_PATH="${MODEL_PATH}" \
TRAIN_DATA="${TRAIN_DATA}" \
VAL_DATA="${VAL_DATA}" \
NGPU=2 \
TP=2 \
FRESH_START=1 \
bash scripts/run_verl_grpo_synthetic_qwen35_9b.sh \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_N}" \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.resume_mode="${RESUME_MODE}"

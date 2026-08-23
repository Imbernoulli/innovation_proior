#!/usr/bin/env bash
# SMOKE GRPO training for the NEW ThetaEvolve + TTT-Discover tasks (thinking ON).
#
# Purpose: validate that the new in-task 0-100 rewards
#   (thetaevolve_ac3, thetaevolve_circle, ttt_erdos)
# compute correctly during a real VERL rollout. A few steps only.
#
# These tasks use OFFLINE numpy evaluators (no FrontierCS/ALE judge), so this
# script does NOT start any judge. It uses its OWN checkpoint/rollout/project
# namespace and its OWN data, so it never collides with the fcs+ale runs or the
# other agent's training dirs.
#
# Submit:
#   sbatch slurm/cc_tt_smoke_thetaevolve_ttt_ailab.sh

#SBATCH --job-name=cc-tt-smoke
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
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
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-local}}"
export TMPDIR="/tmp"

# Let the new reward functions import the ThetaEvolve openevolve package in their
# evaluator subprocess. (TTT erdos is pure numpy, no repo import needed.)
export THETAEVOLVE_ROOT="${THETAEVOLVE_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted}"

# Own namespace — must not collide with fcs+ale runs.
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/cc_tt_thetaevolve_ttt_smoke/qwen35_9b_grpo_tt_smoke}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/cc_tt_rollout_thetaevolve_ttt_smoke}"
export PROJECT_NAME="${PROJECT_NAME:-cc_tt_thetaevolve_ttt_smoke}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_tt_smoke}"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

# Smoke: few steps, validate before train so rewards show up immediately, no in-loop saving spam.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-100000}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VAL_N="${VAL_N:-1}"

# New-task data only. Point ALEBENCH_VAL_DATA at a nonexistent file so the
# underlying script skips ALE validation (and never needs a judge).
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/thetaevolve_ttt/train_thetaevolve_ttt_smoke.parquet}"
export VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/thetaevolve_ttt/val_thetaevolve_ttt.parquet}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/thetaevolve_ttt/__no_alebench__.parquet}"

# 1 GPU smoke layout (8 CPU/GPU cap on ailab).
export NGPU="${NGPU:-1}"
export TP="${TP:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-26624}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-26624}"
export GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
export FRESH_START="${FRESH_START:-1}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}" \
bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  ++data.apply_chat_template_kwargs.enable_thinking=True \
  "$@"

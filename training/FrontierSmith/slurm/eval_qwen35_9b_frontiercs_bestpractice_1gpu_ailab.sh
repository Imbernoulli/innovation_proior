#!/usr/bin/env bash
# FrontierCS Qwen3.5-9B base eval using Qwen3.5 thinking-general sampling plus
# the README Best Practices long-output setting for complex programming tasks.
#
# Submit with:
#   sbatch slurm/eval_qwen35_9b_frontiercs_bestpractice_1gpu_ailab.sh

#SBATCH --job-name=fs-q35-fcsbp
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export SOURCE="${SOURCE:-frontiercs}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/frontiercs_eval_qwen35_9b_qwen_bestpractice_82k_c24_vllm}"
export SAMPLES_JSONL="${SAMPLES_JSONL:-$OUTPUT_DIR/samples_compact.jsonl}"
export SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_DIR/summary.json}"
export N_SAMPLES="${N_SAMPLES:-5}"

# Qwen3.5 README: thinking-general sampling.
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0.0}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# Qwen3.5 README Best Practices: 81,920 output tokens for highly complex
# math/programming competitions and at least 128K context to preserve thinking.
export MAX_TOKENS="${MAX_TOKENS:-81920}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export CONCURRENCY="${CONCURRENCY:-24}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"

export FRONTIERCS_PROMPT_SOURCE="${FRONTIERCS_PROMPT_SOURCE:-official}"
export FRONTIERCS_SCORE_BACKEND="${FRONTIERCS_SCORE_BACKEND:-official}"
export SAVE_TEXT="${SAVE_TEXT:-0}"
export TEXT_PREVIEW_CHARS="${TEXT_PREVIEW_CHARS:-0}"
export SEED="${SEED:-42}"

export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
export TP="${TP:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"

exec bash slurm/eval_qwen35_9b_base_vllm_ailab.sh "$@"

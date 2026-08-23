#!/usr/bin/env bash
# Short FrontierCS sampling diagnostic for Qwen3.5-9B. Saves raw text so we can
# inspect thinking-mode repetition before launching a full 860-sample eval.
#
# Submit with:
#   sbatch slurm/eval_qwen35_9b_frontiercs_diag_sampling_1gpu_ailab.sh

#SBATCH --job-name=fs-q35-fcsdiag
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
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

PROFILE="${PROFILE:-qwen_coding_thinking}"

export SOURCE=frontiercs
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/frontiercs_diag_${PROFILE}}"
export SAMPLES_JSONL="${SAMPLES_JSONL:-$OUTPUT_DIR/samples.jsonl}"
export SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_DIR/summary.json}"
export N_SAMPLES="${N_SAMPLES:-1}"
export FRONTIERCS_LIMIT="${FRONTIERCS_LIMIT:-32}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export CONCURRENCY="${CONCURRENCY:-32}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export FRONTIERCS_PROMPT_SOURCE=official
export FRONTIERCS_SCORE_BACKEND=official
export SAVE_TEXT="${SAVE_TEXT:-1}"
export TEXT_PREVIEW_CHARS="${TEXT_PREVIEW_CHARS:-0}"
export RESUME="${RESUME:-0}"

case "$PROFILE" in
  qwen_coding_thinking)
    export TEMPERATURE="${TEMPERATURE:-0.6}"
    export TOP_P="${TOP_P:-0.95}"
    export TOP_K="${TOP_K:-20}"
    export MIN_P="${MIN_P:-0.0}"
    export PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
    export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
    ;;
  qwen_coding_presence)
    export TEMPERATURE="${TEMPERATURE:-0.6}"
    export TOP_P="${TOP_P:-0.95}"
    export TOP_K="${TOP_K:-20}"
    export MIN_P="${MIN_P:-0.0}"
    export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
    export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
    ;;
  qwen_general_thinking)
    export TEMPERATURE="${TEMPERATURE:-1.0}"
    export TOP_P="${TOP_P:-0.95}"
    export TOP_K="${TOP_K:-20}"
    export MIN_P="${MIN_P:-0.0}"
    export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
    export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
    ;;
  paper_minimal)
    export TEMPERATURE="${TEMPERATURE:-1.0}"
    export TOP_P="${TOP_P:-1.0}"
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE" >&2
    exit 1
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"

exec bash slurm/eval_qwen35_9b_base_vllm_ailab.sh "$@"

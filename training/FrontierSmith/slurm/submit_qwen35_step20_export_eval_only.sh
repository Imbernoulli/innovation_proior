#!/usr/bin/env bash
# Export an already-finished Qwen3.5 mixed-public checkpoint and evaluate it.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public}"
HF_DIR="${HF_DIR:-$PROJECT_ROOT/models/cc_qwen35_9b_mixed_hf_step20}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_qwen35_9b_mixed_step20_thinking_general_32k_both_vllm}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-vllm023}"

export_text=$(
  sbatch \
    --export="ALL,CHECKPOINT_ROOT=$CKPT_DIR,OUTPUT_DIR=$HF_DIR,VENV_DIR=$VENV_DIR" \
    slurm/export_verl_ckpt_to_hf_cpu.sh
)
export_job="${export_text##* }"
echo "export job: $export_job"

eval_text=$(
  sbatch --dependency="afterok:${export_job}" \
    --export="ALL,MODEL_PATH=$HF_DIR,SERVED_MODEL_NAME=cc-qwen35-9b-mixed-step20,OUTPUT_BASE=$OUTPUT_BASE" \
    slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh
)
eval_job="${eval_text##* }"
echo "eval array job: $eval_job"

summary_text=$(
  sbatch --dependency="afterany:${eval_job}" \
    --export="ALL,OUTPUT_BASE=$OUTPUT_BASE" \
    slurm/summarize_qwen35_9b_mixed_thinking_both_cpu.sh
)
summary_job="${summary_text##* }"
echo "summary job: $summary_job"

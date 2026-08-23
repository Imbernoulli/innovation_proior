#!/usr/bin/env bash
# Continue Qwen3.5-9B mixed-public GRPO from the latest checkpoint, then export
# and evaluate the resulting HF model on FrontierCS + ALE-Bench with thinking.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public}"
HF_DIR="${HF_DIR:-$PROJECT_ROOT/models/cc_qwen35_9b_mixed_hf_step20}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_qwen35_9b_mixed_step20_thinking_general_32k_both_vllm}"

train_text=$(
  sbatch --time="${TRAIN_TIME:-02:30:00}" \
    --export="ALL,CKPT_DIR=$CKPT_DIR,FRESH_START=0,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20},SAVE_FREQ=${SAVE_FREQ:-5},TEST_FREQ=${TEST_FREQ:-25},ROLLOUT_N=${ROLLOUT_N:-4},MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192},MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480},MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}" \
    slurm/train_qwen35_9b_mixed_public_ailab.sh
)
train_job="${train_text##* }"
echo "train job: $train_job"

export_text=$(
  sbatch --dependency="afterok:${train_job}" \
    --export="ALL,CHECKPOINT_ROOT=$CKPT_DIR,OUTPUT_DIR=$HF_DIR" \
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
  sbatch --dependency="afterok:${eval_job}" \
    --export="ALL,OUTPUT_BASE=$OUTPUT_BASE" \
    slurm/summarize_qwen35_9b_mixed_thinking_both_cpu.sh
)
summary_job="${summary_text##* }"
echo "summary job: $summary_job"

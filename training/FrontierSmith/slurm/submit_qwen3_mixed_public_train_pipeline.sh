#!/usr/bin/env bash
# Submit Qwen3-8B mixed-public GRPO train/export/eval.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-8B}"
MODEL_TAG="${MODEL_TAG:-qwen3_8b}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/verl_frontiercs_${MODEL_TAG}_mixed_public/${MODEL_TAG}_grpo_mixed_public}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rollout_data_${MODEL_TAG}_mixed_public}"
HF_DIR="${HF_DIR:-$PROJECT_ROOT/models/${MODEL_TAG}_mixed_public_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_${MODEL_TAG}_mixed_public_thinking_general_32k_both_vllm}"
DEPENDENCY="${DEPENDENCY:-}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterok}"

dep_arg=()
if [ -n "$DEPENDENCY" ]; then
  dep_arg=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY}")
fi

train_text=$(
  sbatch "${dep_arg[@]}" \
    --export="ALL,MODEL_PATH=$MODEL_PATH,CKPT_DIR=$CKPT_DIR,ROLLOUT_DIR=$ROLLOUT_DIR,PROJECT_NAME=verl_frontiercs_${MODEL_TAG}_mixed_public,EXPERIMENT_NAME=${MODEL_TAG}_grpo_mixed_public,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-30},SAVE_FREQ=${SAVE_FREQ:-5},TEST_FREQ=${TEST_FREQ:-25},ROLLOUT_N=${ROLLOUT_N:-4}" \
    slurm/train_qwen3_8b_mixed_public_ailab.sh
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
    --export="ALL,MODEL_PATH=$HF_DIR,MODEL_TAG=${MODEL_TAG}_mixed_public,SERVED_MODEL_NAME=${MODEL_TAG}_mixed_public,OUTPUT_DIR=$OUTPUT_DIR" \
    slurm/eval_qwen3_both_thinking_1gpu_ailab.sh
)
eval_job="${eval_text##* }"
echo "trained eval job: $eval_job"

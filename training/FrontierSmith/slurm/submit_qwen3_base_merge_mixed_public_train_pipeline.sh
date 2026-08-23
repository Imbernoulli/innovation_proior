#!/usr/bin/env bash
# Submit mixed-public GRPO train/export/trained-eval pipelines for Qwen3-8B-Base
# and Qwen3-8B/Qwen3-8B-Base linear merge models.
#
# Qwen3-8B itself is handled by slurm/train_qwen3_8b_mixed_public_ailab.sh in
# the baseline chain, so this submitter covers the remaining endpoint/ratios.
#
# Optional:
#   DEPENDENCY="9979417:9979418:9979419:9979420:9979421" \
#     bash slurm/submit_qwen3_base_merge_mixed_public_train_pipeline.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

DEPENDENCY="${DEPENDENCY:-}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterok}"

models=(
  "qwen3_8b_base:$PROJECT_ROOT/models/Qwen3-8B-Base"
  "qwen3_8b_alpha0p25:$PROJECT_ROOT/models/Qwen3-8B-linear-alpha-0p25"
  "qwen3_8b_alpha0p50:$PROJECT_ROOT/models/Qwen3-8B-linear-alpha-0p50"
  "qwen3_8b_alpha0p75:$PROJECT_ROOT/models/Qwen3-8B-linear-alpha-0p75"
)

prev_job="$DEPENDENCY"
for spec in "${models[@]}"; do
  tag="${spec%%:*}"
  model_path="${spec#*:}"

  if [ "${SKIP_QWEN3_BASE:-0}" = "1" ] && [ "$tag" = "qwen3_8b_base" ]; then
    echo "Skipping $tag: SKIP_QWEN3_BASE=1"
    continue
  fi
  if [ "${SKIP_QWEN3_ALPHA0P25:-0}" = "1" ] && [ "$tag" = "qwen3_8b_alpha0p25" ]; then
    echo "Skipping $tag: SKIP_QWEN3_ALPHA0P25=1"
    continue
  fi
  if [ "${SKIP_QWEN3_ALPHA0P50:-0}" = "1" ] && [ "$tag" = "qwen3_8b_alpha0p50" ]; then
    echo "Skipping $tag: SKIP_QWEN3_ALPHA0P50=1"
    continue
  fi
  if [ "${SKIP_QWEN3_ALPHA0P75:-0}" = "1" ] && [ "$tag" = "qwen3_8b_alpha0p75" ]; then
    echo "Skipping $tag: SKIP_QWEN3_ALPHA0P75=1"
    continue
  fi

  if [ ! -d "$model_path" ]; then
    echo "Skipping $tag: missing $model_path" >&2
    continue
  fi

  dep_arg=()
  if [ -n "$prev_job" ]; then
    dep_arg=(--dependency="${DEPENDENCY_TYPE}:${prev_job}")
  fi

  ckpt_dir="$PROJECT_ROOT/checkpoints/verl_frontiercs_${tag}_mixed_public/${tag}_grpo_mixed_public"
  rollout_dir="$PROJECT_ROOT/outputs/rollout_data_${tag}_mixed_public"
  hf_dir="$PROJECT_ROOT/models/${tag}_mixed_public_hf"
  trained_eval_dir="$PROJECT_ROOT/outputs/eval_${tag}_mixed_public_thinking_general_both_vllm"

  echo "Submitting mixed-public train/export/eval pipeline for $tag"
  train_text=$(
    sbatch "${dep_arg[@]}" \
      --export="ALL,MODEL_PATH=$model_path,CKPT_DIR=$ckpt_dir,ROLLOUT_DIR=$rollout_dir,PROJECT_NAME=verl_frontiercs_${tag}_mixed_public,EXPERIMENT_NAME=${tag}_grpo_mixed_public,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-30},SAVE_FREQ=${SAVE_FREQ:-5},TEST_FREQ=${TEST_FREQ:-25},ROLLOUT_N=${ROLLOUT_N:-4}" \
      slurm/train_qwen3_8b_mixed_public_ailab.sh
  )
  train_job="${train_text##* }"
  echo "  train job: $train_job"

  export_text=$(
    sbatch --dependency="afterok:${train_job}" \
      --export="ALL,CHECKPOINT_ROOT=$ckpt_dir,OUTPUT_DIR=$hf_dir" \
      slurm/export_verl_ckpt_to_hf_cpu.sh
  )
  export_job="${export_text##* }"
  echo "  export job: $export_job"

  eval_text=$(
    sbatch --dependency="afterok:${export_job}" \
      --export="ALL,MODEL_PATH=$hf_dir,MODEL_TAG=${tag}_mixed_public,SERVED_MODEL_NAME=${tag}_mixed_public,OUTPUT_DIR=$trained_eval_dir" \
      slurm/eval_qwen3_both_thinking_1gpu_ailab.sh
  )
  eval_job="${eval_text##* }"
  echo "  trained eval job: $eval_job"

  prev_job="$eval_job"
done

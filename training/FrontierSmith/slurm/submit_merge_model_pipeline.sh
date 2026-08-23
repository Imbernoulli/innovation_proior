#!/usr/bin/env bash
# Submit FrontierSmith training + eval pipeline for merged Qwen3-8B / Qwen3-8B-Base models.
#
# Usage:
#   bash slurm/submit_merge_model_pipeline.sh
#
# This submits, for each merge alpha:
#   1. base-model eval
#   2. short GRPO training
#   3. export trained checkpoint to HF
#   4. trained-model eval
#
# Jobs are chained so that at most one GPU job runs per alpha, avoiding
# concurrency surprises on the ailab partition.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs checkpoints outputs/eval

# Dependency on the baseline smoke job (optional). If set, no merge job starts
# until this job completes successfully.
BASELINE_SMOKE_JOB="${BASELINE_SMOKE_JOB:-}"

for alpha in 0.25 0.50 0.75; do
  tag="$(printf '%.2f' "$alpha" | tr '.' 'p')"
  model_name="Qwen3-8B-linear-alpha-${tag}"
  model_path="${PROJECT_ROOT}/models/${model_name}"

  if [ ! -d "$model_path" ]; then
    echo "Skipping $model_name: $model_path not found" >&2
    continue
  fi

  echo "Submitting pipeline for $model_name"

  dep_arg=""
  if [ -n "$BASELINE_SMOKE_JOB" ]; then
    dep_arg="--dependency=afterok:${BASELINE_SMOKE_JOB}"
  fi

  # 1. Base eval
  base_eval_job=$(
    MODEL_PATH="$model_path" \
    EVAL_NAME="${model_name}_base" \
    DATA_PATH="${PROJECT_ROOT}/data/frontiercs/train_synthetic.parquet" \
    MAX_TOKENS=256 \
    N_SAMPLES=1 \
    CONCURRENCY=2 \
    sbatch ${dep_arg} slurm/eval_qwen3_model_smoke_ailab.sh
  )
  base_eval_job="${base_eval_job##* }"
  echo "  base eval job: $base_eval_job"

  # 2. Short training
  train_job=$(
    MODEL_PATH="$model_path" \
    MODEL_RUN_NAME="${model_name}_grpo_smoke" \
    CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${model_name}_smoke" \
    ROLLOUT_DIR="${PROJECT_ROOT}/outputs/rollout_data_${model_name}_smoke" \
    EXPERIMENT_NAME="${model_name}_grpo_smoke_1gpu" \
    sbatch --dependency=afterok:"${base_eval_job}" slurm/train_qwen3_8b_smoke_1gpu_ailab.sh
  )
  train_job="${train_job##* }"
  echo "  train job: $train_job"

  # 3. Export checkpoint to HF (CPU job)
  export_job=$(
    CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${model_name}_smoke" \
    OUTPUT_DIR="${PROJECT_ROOT}/models/${model_name}_trained_smoke_hf" \
    sbatch --dependency=afterok:"${train_job}" slurm/export_verl_ckpt_to_hf_cpu.sh
  )
  export_job="${export_job##* }"
  echo "  export job: $export_job"

  # 4. Trained-model eval
  trained_eval_job=$(
    MODEL_PATH="${PROJECT_ROOT}/models/${model_name}_trained_smoke_hf" \
    EVAL_NAME="${model_name}_trained" \
    DATA_PATH="${PROJECT_ROOT}/data/frontiercs/train_synthetic.parquet" \
    MAX_TOKENS=256 \
    N_SAMPLES=1 \
    CONCURRENCY=2 \
    sbatch --dependency=afterok:"${export_job}" slurm/eval_qwen3_model_smoke_ailab.sh
  )
  trained_eval_job="${trained_eval_job##* }"
  echo "  trained eval job: $trained_eval_job"
done

#!/usr/bin/env bash
# Submit the FrontierSmith Qwen3-8B baseline pipeline.
#
# Usage:
#   bash slurm/submit_baseline_pipeline.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs checkpoints outputs/eval

MODEL_PATH="${PROJECT_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/frontiercs/train_synthetic.parquet}"

echo "Submitting baseline pipeline for Qwen3-8B"

# 1. Smoke training (1 GPU, 30 min)
smoke_job=$(sbatch slurm/train_qwen3_8b_smoke_1gpu_ailab.sh)
smoke_job="${smoke_job##* }"
echo "  smoke job: $smoke_job"

# 2. Short training (1 GPU, 30 min)
train_job=$(
  MODEL_RUN_NAME="Qwen3-8B_grpo_baseline" \
  CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/Qwen3-8B_baseline" \
  ROLLOUT_DIR="${PROJECT_ROOT}/outputs/rollout_data_Qwen3-8B_baseline" \
  EXPERIMENT_NAME="Qwen3-8B_grpo_baseline_1gpu" \
  sbatch --dependency=afterok:"${smoke_job}" slurm/train_qwen3_8b_smoke_1gpu_ailab.sh
)
train_job="${train_job##* }"
echo "  train job: $train_job"

# 3. Base model eval (1 GPU, 1 hr)
base_eval_job=$(
  MODEL_PATH="$MODEL_PATH" \
  EVAL_NAME="Qwen3-8B_base" \
  DATA_PATH="$DATA_PATH" \
  MAX_TOKENS=256 \
  N_SAMPLES=1 \
  CONCURRENCY=2 \
  sbatch --dependency=afterok:"${smoke_job}" slurm/eval_qwen3_model_smoke_ailab.sh
)
base_eval_job="${base_eval_job##* }"
echo "  base eval job: $base_eval_job"

# 4. Export trained checkpoint to HF (CPU, 2 hr)
export_job=$(
  CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/Qwen3-8B_baseline" \
  OUTPUT_DIR="${PROJECT_ROOT}/models/Qwen3-8B_baseline_trained_smoke_hf" \
  sbatch --dependency=afterok:"${train_job}" slurm/export_verl_ckpt_to_hf_cpu.sh
)
export_job="${export_job##* }"
echo "  export job: $export_job"

# 5. Trained model eval (1 GPU, 1 hr)
trained_eval_job=$(
  MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-8B_baseline_trained_smoke_hf" \
  EVAL_NAME="Qwen3-8B_trained" \
  DATA_PATH="$DATA_PATH" \
  MAX_TOKENS=256 \
  N_SAMPLES=1 \
  CONCURRENCY=2 \
  sbatch --dependency=afterok:"${export_job}" slurm/eval_qwen3_model_smoke_ailab.sh
)
trained_eval_job="${trained_eval_job##* }"
echo "  trained eval job: $trained_eval_job"

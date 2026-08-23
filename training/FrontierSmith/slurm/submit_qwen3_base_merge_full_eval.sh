#!/usr/bin/env bash
# Submit full FrontierCS + ALE-Bench eval for Qwen3-8B endpoints and merge ratios.
#
# Defaults submit five independent 1-GPU jobs:
#   Qwen3-8B, Qwen3-8B-Base, alpha 0.25, alpha 0.50, alpha 0.75.
#
# Optional:
#   DEPENDENCY=9979106 bash slurm/submit_qwen3_base_merge_full_eval.sh
#   MAX_PARALLEL=2 bash slurm/submit_qwen3_base_merge_full_eval.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

DEPENDENCY="${DEPENDENCY:-}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

models=(
  "qwen3_8b:$PROJECT_ROOT/models/Qwen3-8B"
  "qwen3_8b_base:$PROJECT_ROOT/models/Qwen3-8B-Base"
  "qwen3_8b_alpha0p25:$PROJECT_ROOT/models/Qwen3-8B-linear-alpha-0p25"
  "qwen3_8b_alpha0p50:$PROJECT_ROOT/models/Qwen3-8B-linear-alpha-0p50"
  "qwen3_8b_alpha0p75:$PROJECT_ROOT/models/Qwen3-8B-linear-alpha-0p75"
)

submitted_jobs=()
for idx in "${!models[@]}"; do
  spec="${models[$idx]}"
  tag="${spec%%:*}"
  model_path="${spec#*:}"

  if [ ! -d "$model_path" ]; then
    echo "Skipping $tag: missing $model_path" >&2
    continue
  fi

  dep_arg=()
  deps=()
  if [ -n "$DEPENDENCY" ]; then
    deps+=("$DEPENDENCY")
  fi
  back_idx=$((idx - MAX_PARALLEL))
  if [ "$back_idx" -ge 0 ]; then
    deps+=("${submitted_jobs[$back_idx]}")
  fi
  if [ "${#deps[@]}" -gt 0 ]; then
    IFS=:
    dep_arg=(--dependency="afterany:${deps[*]}")
    unset IFS
  fi

  echo "Submitting full eval for $tag"
  job_text=$(
    sbatch "${dep_arg[@]}" \
      --export="ALL,MODEL_PATH=$model_path,MODEL_TAG=$tag,SERVED_MODEL_NAME=$tag,PORT_OFFSET=$((idx * 10)),OUTPUT_DIR=$PROJECT_ROOT/outputs/eval_${tag}_thinking_general_both_vllm" \
      slurm/eval_qwen3_both_thinking_1gpu_ailab.sh
  )
  job_id="${job_text##* }"
  echo "  job: $job_id"
  submitted_jobs+=("$job_id")
done

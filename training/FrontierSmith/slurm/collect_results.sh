#!/usr/bin/env bash
# Collect results from completed baseline and merge-model jobs.
# Usage: bash slurm/collect_results.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_FILE="outputs/eval_summary.csv"
mkdir -p outputs/eval

{
  echo "model,metric,value"

  # Baseline base model eval
  for csv in outputs/eval/Qwen3-8B_base_smoke.csv outputs/eval/Qwen3-8B_base_smoke.csv; do
    if [ -f "$csv" ]; then
      awk -F, 'NR>1 {print "Qwen3-8B-base," $1 "," $2}' "$csv"
    fi
  done

  # Baseline trained model eval
  for csv in outputs/eval/Qwen3-8B_trained_smoke.csv outputs/eval/Qwen3-8B_trained_smoke.csv; do
    if [ -f "$csv" ]; then
      awk -F, 'NR>1 {print "Qwen3-8B-trained," $1 "," $2}' "$csv"
    fi
  done

  # Merge models
  for alpha in 0.25 0.50 0.75; do
    tag="$(printf '%.2f' "$alpha" | tr '.' 'p')"
    model_name="Qwen3-8B-linear-alpha-${tag}"

    for csv in "outputs/eval/${model_name}_base_smoke.csv" "outputs/eval/${model_name}_base_smoke.csv"; do
      if [ -f "$csv" ]; then
        awk -F, -v m="${model_name}-base" 'NR>1 {print m "," $1 "," $2}' "$csv"
      fi
    done

    for csv in "outputs/eval/${model_name}_trained_smoke.csv" "outputs/eval/${model_name}_trained_smoke.csv"; do
      if [ -f "$csv" ]; then
        awk -F, -v m="${model_name}-trained" 'NR>1 {print m "," $1 "," $2}' "$csv"
      fi
    done
  done
} > "$RESULTS_FILE"

echo "Wrote summary to $RESULTS_FILE"
cat "$RESULTS_FILE"

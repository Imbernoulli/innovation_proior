#!/usr/bin/env bash
# Merge sharded Qwen3.5-9B mixed thinking eval JSONL files into one summary.

#SBATCH --job-name=fs-q35mix-sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
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

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm}"
N_SAMPLES="${N_SAMPLES:-5}"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/summarize_base_eval_hf.py \
  "$OUTPUT_BASE/shards/shard_*/samples.jsonl" \
  --n-samples "$N_SAMPLES" \
  --output-json "$OUTPUT_BASE/summary.json"

python scripts/summarize_vllm_eval_summary.py "$OUTPUT_BASE/summary.json"

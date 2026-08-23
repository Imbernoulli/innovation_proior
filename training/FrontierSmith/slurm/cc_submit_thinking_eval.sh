#!/usr/bin/env bash
# Reusable submitter: evaluate a HF model on FrontierCS + ALE in THINKING 32K口径
# (official prompt + official scorer), as a SINGLE 1-GPU job over ALL problems,
# then summarize. Single-shard avoids the multi-shard co-location vLLM-init crashes.
#
# Usage:   slurm/cc_submit_thinking_eval.sh <MODEL_DIR> <TAG>
# Output:  outputs/cc_eval_<TAG>_thinking_general_32k_both_vllm/{shards/shard_0,summary.json}

set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT"
MODEL_DIR="${1:?need MODEL_DIR}"
TAG="${2:?need TAG}"
OB="$ROOT/outputs/cc_eval_${TAG}_thinking_general_32k_both_vllm"

if [ ! -e "$MODEL_DIR/config.json" ]; then echo "ERROR: $MODEL_DIR has no config.json" >&2; exit 1; fi

# Single job: NUM_SHARDS=1 SHARD_IDX=0 -> all 182 problems in one vLLM. Higher
# concurrency to offset doing the whole set on one GPU (qwen3_5 linear attn = small KV).
arr=$(sbatch --parsable --job-name="cc-think-${TAG}" \
  --export=ALL,MODEL_PATH="$MODEL_DIR",OUTPUT_BASE="$OB",SERVED_MODEL_NAME="cc-${TAG}-think",ENABLE_THINKING=1 \
  slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh)

sumj=$(sbatch --parsable --dependency=afterany:$arr \
  --job-name="cc-sum-${TAG}" \
  --export=ALL,OUTPUT_BASE="$OB",N_SAMPLES=5 \
  slurm/summarize_qwen35_9b_mixed_thinking_both_cpu.sh)

echo "TAG=$TAG MODEL=$MODEL_DIR"
echo "  eval_job=$arr summarize=$sumj  output=$OB/summary.json"

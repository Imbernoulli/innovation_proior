#!/usr/bin/env bash
# Single-command submitter for the reusable thinking-mode FrontierCS+ALE eval.
#
# Usage:
#   slurm/cc_submit_eval.sh <MODEL_DIR> <TAG> [VALIDATE]
#
#   <MODEL_DIR>  local HF model dir (must contain config.json)
#   <TAG>        short label; output -> outputs/cc_eval_<TAG>_thinking_32k_both_vllm/summary.json
#   VALIDATE     optional; pass "1" for a short (<=1h) sanity job (few problems)
#
# Examples:
#   # full eval (all 172 FCS + 10 ALE problems, 5 samples, 32K thinking)
#   slurm/cc_submit_eval.sh /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B base_q35_9b
#   # short validation
#   slurm/cc_submit_eval.sh .../models/Qwen3-8B val_q3_8b 1

set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT"

MODEL_DIR="${1:?need MODEL_DIR}"
TAG="${2:?need TAG}"
VALIDATE="${3:-0}"

if [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "ERROR: $MODEL_DIR has no config.json" >&2
  exit 1
fi

# Validation jobs are short (1h, capped problems); full jobs get up to 6h.
if [ "$VALIDATE" = "1" ]; then
  TIME="01:00:00"; JOBNAME="cc-eval-val-${TAG}"
else
  TIME="06:00:00"; JOBNAME="cc-eval-${TAG}"
fi

JID=$(sbatch --parsable \
  --job-name="$JOBNAME" \
  --time="$TIME" \
  --export=ALL,MODEL_PATH="$MODEL_DIR",TAG="$TAG",VALIDATE="$VALIDATE",N_SAMPLES="${N_SAMPLES:-5}" \
  slurm/cc_eval_thinking_both_ailab.sh)

echo "submitted job=$JID  TAG=$TAG  MODEL=$MODEL_DIR  VALIDATE=$VALIDATE"
echo "  log:     logs/${JOBNAME}-${JID}.out"
echo "  summary: outputs/cc_eval_${TAG}_thinking_32k_both_vllm/summary.json"

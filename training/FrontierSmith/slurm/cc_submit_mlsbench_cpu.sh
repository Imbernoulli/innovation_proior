#!/usr/bin/env bash
# Single-command submitter for the MLS-Bench CPU-task eval (vLLM-served model).
#
# Usage:
#   slurm/cc_submit_mlsbench_cpu.sh <MODEL_DIR> <TAG> [SMOKE_TASK]
#
#   <MODEL_DIR>   local HF model dir (must contain config.json)
#   <TAG>         short label; output -> outputs/cc_mlsbench_cpu_<TAG>/summary.json
#   SMOKE_TASK    optional; if given, run ONLY that one task (smoke) on a short job
#
# Examples:
#   # full 20-task CPU eval
#   slurm/cc_submit_mlsbench_cpu.sh /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3-8B q3_8b
#   # 1-task smoke
#   slurm/cc_submit_mlsbench_cpu.sh .../models/Qwen3-8B q3_8b ml-clustering-algorithm

set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT"

MODEL_DIR="${1:?need MODEL_DIR}"
TAG="${2:?need TAG}"
SMOKE_TASK="${3:-}"

if [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "ERROR: $MODEL_DIR has no config.json" >&2
  exit 1
fi

EXPORTS="ALL,MODEL_PATH=${MODEL_DIR},TAG=${TAG}"
if [ -n "$SMOKE_TASK" ]; then
  TIME="01:30:00"; JOBNAME="cc-mlsbench-cpu-smoke-${TAG}"
  EXPORTS="${EXPORTS},SMOKE_TASK=${SMOKE_TASK},CONCURRENCY=1"
else
  TIME="08:00:00"; JOBNAME="cc-mlsbench-cpu-${TAG}"
fi

JID=$(sbatch --parsable \
  --job-name="$JOBNAME" \
  --time="$TIME" \
  --export="$EXPORTS" \
  slurm/cc_eval_mlsbench_cpu_ailab.sh)

echo "submitted job=$JID  TAG=$TAG  MODEL=$MODEL_DIR  SMOKE_TASK=${SMOKE_TASK:-<none>}"
echo "  log:     logs/${JOBNAME}-${JID}.out"
echo "  summary: outputs/cc_mlsbench_cpu_${TAG}/summary.json"

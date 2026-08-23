#!/usr/bin/env bash
# Submit helper for cc_rl_mlsbench.sh (mirrors cc_rl_frontiersmith_synth_submit.sh).
# Scales CPUS=GPUS*8, MEM=GPUS*120G per the ailab per-GPU caps.
#
# Usage:
#   scripts/cc_rl_mlsbench_submit.sh [EXPERIMENT_NAME] [GPUS] [STEPS]
# Env passthrough: MODEL_PATH, TRAIN_DATA, MLS_RL_* knobs, TRAIN_BATCH_SIZE, ROLLOUT_N ...
set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT"

EXPERIMENT_NAME="${1:-${EXPERIMENT_NAME:-mlsrl_qwen35_9b_grpo_smoke}}"
GPUS="${2:-${NGPU:-2}}"
STEPS="${3:-${TOTAL_TRAINING_STEPS:-3}}"

CPUS=$(( GPUS * 8 ))
MEMG=$(( GPUS * 120 ))

JID=$(sbatch --parsable \
  --job-name="rl-mls-${EXPERIMENT_NAME}" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="$CPUS" \
  --mem="${MEMG}G" \
  --export="ALL,EXPERIMENT_NAME=${EXPERIMENT_NAME},NGPU=${GPUS},TOTAL_TRAINING_STEPS=${STEPS}" \
  slurm/cc_rl_mlsbench.sh)

echo "submitted job=$JID EXPERIMENT_NAME=$EXPERIMENT_NAME GPUS=$GPUS STEPS=$STEPS"
echo "  log:  logs/rl-mls-${EXPERIMENT_NAME}-${JID}.out"
echo "  ckpt: checkpoints/rl_mlsbench/${EXPERIMENT_NAME}/"
echo "  episodes: outputs/mls_rl/${EXPERIMENT_NAME}/episode_logs/"

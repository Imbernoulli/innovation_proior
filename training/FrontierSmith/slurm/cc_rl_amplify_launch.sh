#!/usr/bin/env bash
# cc_rl_amplify_launch.sh  (OWN namespace: rl_*/cc_rl_amplify — does NOT touch any
# other agent's namespaces or the protected orchestrator/results files)
#
# Helper to launch ONE FrontierCS+ALE-Bench per-task-NORMALIZED GRPO RL run on
# 2 GPUs, starting from a given model, using the validated 2-GPU config
# (MAX_RESPONSE_LENGTH=16000, gpu_memory_utilization=0.45, expandable_segments).
#
# Usage:
#   slurm/cc_rl_amplify_launch.sh <TAG> <MODEL_PATH> [STEPS] [extra verl overrides...]
# Default STEPS = recipe default (60).
set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
DATA=$ROOT/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet

TAG="$1"; MODEL="$2"; STEPS="${3:-60}"
shift || true; shift || true; shift || true || true
EXTRA=("$@")

[ -f "$MODEL/config.json" ] || { echo "ERROR: no config.json in $MODEL" >&2; exit 1; }

sbatch \
  --job-name="rl-amp-${TAG}" \
  --qos=gpu-short \
  --time=23:59:00 \
  --export=ALL,\
TRAIN_DATA=$DATA,\
MODEL_PATH=$MODEL,\
MAX_RESPONSE_LENGTH=16000,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,\
FS_PERTASK_REWARD_NORM=1,\
CKPT_DIR=$ROOT/checkpoints/cc_rl_amplify/${TAG},\
ROLLOUT_DIR=$ROOT/outputs/cc_rl_amplify_rollout/${TAG},\
PROJECT_NAME=cc_rl_amplify,\
EXPERIMENT_NAME=${TAG},\
TOTAL_TRAINING_STEPS=${STEPS} \
  "$ROOT/slurm/cc_tt_train_mixed_pertask_norm_2gpu_ailab.sh" \
  trainer.total_training_steps=${STEPS} "${EXTRA[@]}"

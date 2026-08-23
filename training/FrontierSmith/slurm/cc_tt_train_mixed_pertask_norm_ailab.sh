#!/usr/bin/env bash
# FULL normalized multi-task RL run: Qwen3.5-9B GRPO on the mixed parquet
#   FrontierCS 182 + ALE-Bench full 40 + thetaevolve_circle 24 + thetaevolve_ac3 24
#   + ttt_erdos 24  (=294 rows, data/mixed/train_fcs172_fsmith10_ale40_thetaevolve_ttt72.parquet)
#
# WITH WITHIN-TASK (per-task) reward normalization ENABLED: each task's raw scorer
# score is mapped onto a common [0,1] scale using the task's own reward bounds, so no
# single task's raw scale dominates the GRPO policy gradient. Implemented in
# verl/verl/utils/reward_score/pertask_norm.py and applied in the reward managers;
# gated by FS_PERTASK_REWARD_NORM. Per-task pre/post-normalization reward stats
# (reward_raw / reward_normed) are logged per data_source in the rollout dump.
#
# This mirrors slurm/cc_tt_train_mixed_thetaevolve_ttt_thinking_ailab.sh (same data,
# same thinking-long recipe) but turns ON per-task normalization and uses its OWN
# checkpoint/rollout/project namespace (..._norm) so it never collides with the
# non-normalized cc_tt_* run or the oe-*/om-* SFT/soup jobs.
#
# Submit:
#   sbatch slurm/cc_tt_train_mixed_pertask_norm_ailab.sh

#SBATCH --job-name=cc-tt-mixnorm
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# ---- WITHIN-TASK reward normalization: ON, target [0,1] ----
export FS_PERTASK_REWARD_NORM="${FS_PERTASK_REWARD_NORM:-1}"
export FS_PERTASK_REWARD_NORM_TARGET="${FS_PERTASK_REWARD_NORM_TARGET:-1.0}"

# Point the frontiercs reward at the judge port this recipe starts (8092). Without
# this the reward worker connects to the legacy hard-pinned :8082 -> JudgeInfraError.
export PORT="${PORT:-8092}"
export FRONTIERCS_JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://localhost:${PORT}}"

# New-task reward subprocess needs to import the OpenEvolve evaluator.
export THETAEVOLVE_ROOT="${THETAEVOLVE_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted}"

# OWN namespace (distinct from cc_tt_mixed_* non-norm run and oe-*/om-* jobs).
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/cc_tt_mixed_pertask_norm/qwen35_9b_grpo_mix_tt_norm}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/cc_tt_rollout_mixed_pertask_norm}"
export PROJECT_NAME="${PROJECT_NAME:-cc_tt_mixed_pertask_norm}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_mix_tt_norm}"

# The mixed parquet that INCLUDES the new tasks.
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_fcs172_fsmith10_ale40_thetaevolve_ttt72.parquet}"

# Delegate to the canonical thinking-long recipe (handles judge, env, GRPO args).
exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@"

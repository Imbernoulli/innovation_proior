#!/usr/bin/env bash
# THINKING-ENABLED, long GRPO for Qwen3.5-9B on the FULL mix INCLUDING the new
# ThetaEvolve + TTT-Discover tasks:
#   FrontierCS 172 + FrontierSmith 10 + ALE-Bench full 40
#   + thetaevolve_ac3 24 + thetaevolve_circle 24 + ttt_erdos 24   (=294 rows)
#
# This mirrors slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh (the canonical
# thinking-long recipe: response 30000, max_model_len 40960, thinking ON) but swaps
# in the mixed parquet that contains the new tasks and adds THETAEVOLVE_ROOT so the
# new reward functions can import the OpenEvolve evaluator in their subprocess.
#
# Because the mix still contains FrontierCS + ALE rows, this launcher starts the
# FrontierCS judge (like the canonical recipe). It uses its OWN
# checkpoint/rollout/project namespace and does NOT touch the other agent's dirs.
#
# Submit:
#   sbatch slurm/cc_tt_train_mixed_thetaevolve_ttt_thinking_ailab.sh

#SBATCH --job-name=cc-tt-mixthink
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

# New-task reward subprocess needs to import the OpenEvolve evaluator.
export THETAEVOLVE_ROOT="${THETAEVOLVE_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted}"

# Own namespace so it never collides with fcs+ale runs.
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/cc_tt_mixed_thetaevolve_ttt_thinking/qwen35_9b_grpo_mix_tt}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/cc_tt_rollout_mixed_thetaevolve_ttt_thinking}"
export PROJECT_NAME="${PROJECT_NAME:-cc_tt_mixed_thetaevolve_ttt_thinking}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_mix_tt}"

# The mixed parquet that INCLUDES the new tasks.
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_fcs172_fsmith10_ale40_thetaevolve_ttt72.parquet}"

# Delegate to the canonical thinking-long recipe (handles judge, env, args).
exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@"

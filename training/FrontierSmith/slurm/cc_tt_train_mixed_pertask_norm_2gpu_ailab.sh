#!/usr/bin/env bash
# FULL normalized multi-task RL run on 2 GPUs (2x H200) -- a 2-GPU variant of
# slurm/cc_tt_train_mixed_pertask_norm_ailab.sh so it schedules on the common
# 2-3 GPU pockets (ailab has no node with 4 free GPUs; the 4-GPU job starved).
#
# Same run in every other respect:
#   * per-task reward normalization ON (FS_PERTASK_REWARD_NORM=1), target [0,1]
#   * FrontierCS judge on :8092 with the judge-port fix (FRONTIERCS_JUDGE_URL)
#   * OWN namespace checkpoints/cc_tt_mixed_pertask_norm/qwen35_9b_grpo_mix_tt_norm
#   * project cc_tt_mixed_pertask_norm, experiment qwen35_9b_grpo_mix_tt_norm
#   * thinking ON, response 30000, max_model_len 40960 (FULL context)
#   * the EXTENDED 5-task parquet (342 rows: frontiercs 182, alebench 40,
#     thetaevolve_circle/ac3 + ttt_erdos/ac1/ac2 @24)
#
# 2-GPU fit (validated by the zz-rl2gpu-smoke full-context smoke):
#   * NGPU=2, TP=1, gres=gpu:2
#   * param/optimizer/ref OFFLOAD stays ON (canonical default) -- mandatory at the
#     40960 context (the prior 4-GPU full run peaked at 136.4 GB reserved/H200 with
#     offload ON).
#   * gpu_memory_utilization 0.6 -> 0.45: shrink the vLLM KV-cache reservation so the
#     larger 2-GPU FSDP gather + 40960-token activations fit in 139 GB.
#   * max_num_seqs 8 -> 4: bound concurrent KV cache on the smaller 2-GPU KV budget.
#   * 480 G host RAM + 16 CPUs (ailab caps 8 cores/GPU).
#   * --qos=gpu-short so it is NOT forced into gpu-test (MaxJobsPU=3, full).
#
# Submit:
#   sbatch slurm/cc_tt_train_mixed_pertask_norm_2gpu_ailab.sh

#SBATCH --job-name=cc-tt-mixnorm-2g
#SBATCH --partition=ailab
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
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

# FrontierCS judge port fix (judge launcher binds 8092, not legacy 8082).
export PORT="${PORT:-8092}"
export FRONTIERCS_JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://localhost:${PORT}}"

# New-task reward subprocess needs the OpenEvolve evaluator.
export THETAEVOLVE_ROOT="${THETAEVOLVE_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted}"

# OWN namespace (identical to the 4-GPU launcher so this is THE normalized run).
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/cc_tt_mixed_pertask_norm/qwen35_9b_grpo_mix_tt_norm}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/cc_tt_rollout_mixed_pertask_norm}"
export PROJECT_NAME="${PROJECT_NAME:-cc_tt_mixed_pertask_norm}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_mix_tt_norm}"

# The EXTENDED 5-task parquet (342 rows).
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_fcs172_fsmith10_ale40_thetaevolve_ttt120_CANDIDATE.parquet}"

# ---- 2-GPU layout + memory fit ----
export NGPU="${NGPU:-2}"
export TP="${TP:-1}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Delegate to the canonical thinking-long recipe (handles judge, env, GRPO args,
# the full 30000/40960 context). The trailing Hydra override shrinks the vLLM KV
# cache so the 2-GPU run fits; offloads stay ON (recipe default).
exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45

#!/usr/bin/env bash
# SMOKE TEST for WITHIN-TASK (per-task) reward normalization on the mixed RL run.
#
# Goal: prove (a) rewards are normalized per task (FS_PERTASK_REWARD_NORM=1 maps each
# task's raw scorer score onto a common [0,1] scale) and (b) GRPO actually steps.
# Tiny by design: 2 GPUs (FSDP shards the actor so offload frees per-GPU memory),
# 2 steps, short walltime, a 10-row mixed parquet
# (frontiercs + thetaevolve_circle/ac3 + ttt_erdos; ALE skipped to stay fast).
#
# It DELEGATES to the canonical thinking-long recipe
# (slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh) so the smoke exercises the
# exact same code path as the full run -- only the env knobs differ.
#
# Uses its OWN checkpoint/rollout/project namespace; never touches oe-*/om-* jobs or
# the other agents' checkpoint dirs.
#
# Submit:
#   sbatch slurm/cc_smoke_pertask_norm_ailab.sh

#SBATCH --job-name=cc-smoke-pertasknorm
#SBATCH --partition=ailab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# ---- per-task reward normalization: ON, target [0,1] ----
export FS_PERTASK_REWARD_NORM=1
export FS_PERTASK_REWARD_NORM_TARGET=1.0

# Point the frontiercs reward at the judge port this recipe actually starts (8092).
# (frontiercs.py defaults to http://localhost:$PORT; set both for safety.)
export PORT="${PORT:-8092}"
export FRONTIERCS_JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://localhost:${PORT}}"

# New-task reward subprocess needs to import the OpenEvolve evaluator.
export THETAEVOLVE_ROOT="${THETAEVOLVE_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted}"

# OWN namespace (distinct from cc_tt_* / cc_verl_* / oe-* / om-*).
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/cc_smoke_pertask_norm/qwen35_9b_grpo_smoke_norm}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/cc_smoke_pertask_norm_rollout}"
export PROJECT_NAME="${PROJECT_NAME:-cc_smoke_pertask_norm}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_smoke_norm}"

# Tiny mixed parquet: frontiercs + thetaevolve + ttt (no ALE, to keep it fast).
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/cc_pertasknorm_smoke.parquet}"
# Validate against the same tiny file; point ALE val at a non-existent path so the
# recipe's "[ALE-Bench] ... not found" branch fires and no ALE val is added.
export VAL_DATA="${VAL_DATA:-$TRAIN_DATA}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/__no_such_val__.parquet}"

# Small + short: 2 GPUs (so the FSDP actor SHARDS across cards and param/optimizer
# offload actually frees per-GPU memory -- a single-GPU NO_SHARD actor gathers the
# full 9.4B params to one card at weight-sync time and OOMs on a shared H200), 2
# steps, few rollouts, skip in-loop validation, save once.
export NGPU="${NGPU:-2}"
export TP="${TP:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export VAL_N="${VAL_N:-1}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export TEST_FREQ="${TEST_FREQ:-100000}"   # skip in-loop validation
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FRESH_START="${FRESH_START:-1}"

# Keep memory modest for a single GPU smoke (still thinking-enabled). Short responses
# so the FSDP actor + vLLM colocated on ONE shared H200 fit alongside any co-tenant.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1500}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-12288}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

# Reduce GPU memory fragmentation so the actor->vLLM IPC weight sync does not OOM on a
# shared card (the prior 0.35 run OOM'd at update_weights_from_ipc because a co-tenant
# held most of the H200).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Delegate to the canonical thinking-long recipe (handles judge, env, GRPO args).
# Hydra overrides appended after "$@" win over the recipe's defaults:
#   * gpu_memory_utilization -> 0.35       (tiny KV cache; leave room for the FSDP actor)
#   * enforce_eager=True                   (skip CUDA-graph capture memory)
#   * param_offload/optimizer_offload=False: keep the 9.4B actor + Adam state on the
#     2x H200 (ample VRAM) instead of offloading to CPU. The offloaded variant was
#     getting host-RAM OOM-killed (Ray "Worker exit ... OOM killer") on the shared
#     node during the long eager-mode rollout. On-GPU weights also generate faster.
exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False

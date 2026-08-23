#!/usr/bin/env bash
# FULL-CONTEXT 2-GPU SMOKE for the per-task-normalized mixed RL run.
#
# Purpose: prove the multi-task GRPO run STEPS on 2x H200 at the FULL 40960 context
# (response 30000, max_model_len 40960, thinking ON) WITHOUT OOM, with per-task
# reward normalization ON. The prior agent already proved 2-GPU steps but only at a
# TRUNCATED 1500-token response (job 10135392). This validates the FULL context.
#
# Sizing reasoning (vs the canonical 4-GPU recipe defaults):
#   * The prior FULL-context 4-GPU run (job 10010261) peaked at max_memory_reserved
#     136.4 GB / H200 with offload ON + gpu_memory_utilization=0.6. On 2 GPUs each
#     FSDP shard (and its gathered-layer + 40960-token activation) is ~2x larger, so
#     gpu_mem_util=0.6 would OOM. -> drop the vLLM KV-cache reservation to 0.45.
#   * Keep param/optimizer/ref OFFLOAD ON (canonical default). At full context the
#     resident actor+Adam state must stay off-GPU; the prior agent only disabled
#     offload for the tiny 1500-token smoke. Request 480G host RAM (the prior smoke
#     host-OOM'd at a 240G ask; the 4-GPU full run used ~350G host).
#   * max_num_seqs=4 (down from 8) bounds concurrent KV cache on the smaller 2-GPU
#     KV budget; max_num_batched_tokens stays 40960 so a single full-context prefill
#     still fits. rollout.n=8 keeps a real GRPO group for the reward signal.
#   * enforce_eager=True skips CUDA-graph capture memory (smoke only; the full run
#     keeps graphs for speed).
#
# OWN namespace + job-name (zz-rl2gpu-smoke); never collides with the matrix.
#
# Submit:
#   sbatch slurm/zz_rl2gpu_smoke_ailab.sh

#SBATCH --job-name=zz-rl2gpu-smoke
#SBATCH --partition=ailab
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
# 01:05:00 so Slurm routes this to the gpu-short QOS (jobs <=1h are forced into
# gpu-test, MaxJobsPU=3, which the SFT/eval matrix already fills). The smoke itself
# runs only 2 steps (~12-15 min) and is cancelled once it has stepped, so actual
# runtime stays well under 50 min.
#SBATCH --time=01:05:00
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

# Point the frontiercs reward at the judge port this recipe starts (8092).
export PORT="${PORT:-8092}"
export FRONTIERCS_JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://localhost:${PORT}}"

# New-task reward subprocess needs the OpenEvolve evaluator.
export THETAEVOLVE_ROOT="${THETAEVOLVE_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted}"

# OWN smoke namespace (distinct from cc_tt_mixed_pertask_norm full run + matrix).
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/zz_rl2gpu_smoke/qwen35_9b_grpo_rl2gpu_smoke}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/zz_rl2gpu_smoke_rollout}"
export PROJECT_NAME="${PROJECT_NAME:-zz_rl2gpu_smoke}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_grpo_rl2gpu_smoke}"

# Tiny multi-task parquet (frontiercs + thetaevolve_circle/ac3 + ttt_erdos; no ALE,
# to keep each step fast) -- but run at the FULL context below.
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/cc_pertasknorm_smoke.parquet}"
export VAL_DATA="${VAL_DATA:-$TRAIN_DATA}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/__no_such_val__.parquet}"

# 2 GPUs, FULL context, 2 steps, skip in-loop val, save once.
export NGPU="${NGPU:-2}"
export TP="${TP:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VAL_N="${VAL_N:-1}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FRESH_START="${FRESH_START:-1}"

# FULL context (matches the real run): thinking room.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-30000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

# Reduce fragmentation so the actor->vLLM IPC weight sync does not OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Delegate to the canonical thinking-long recipe. Hydra overrides after "$@" win:
#   * gpu_memory_utilization 0.6 -> 0.45 (smaller KV cache, room for the 2-GPU gather)
#   * enforce_eager=True             (smoke only; skip CUDA-graph capture memory)
#   * offloads stay ON (recipe defaults) -- mandatory at full context.
exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=True

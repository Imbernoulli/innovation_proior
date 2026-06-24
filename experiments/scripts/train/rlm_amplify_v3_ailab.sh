#!/usr/bin/env bash
# rlm_amplify_v3: FAST GRPO launcher. Same proven harness as v2
# (slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh -> the judge + GRPO recipe),
# but with the SLOWNESS FIXED via config:
#
#   * ACTOR offload OFF: actor.fsdp_config.param_offload=False + optimizer_offload=False.
#     On 4xH200, FSDP already shards optimizer (~27GB/GPU) + params (~4.5GB/GPU) across
#     the GPUs, so full CPU offload was pure overhead. This is the dominant fix:
#     update_actor dropped from ~700-790s to <250s. (REF model stays offloaded -- it is
#     frozen/forward-only, cheap to offload, matches verl stock GPU examples.)
#   * response 30000 -> 20000, max_model_len 40960 -> 30720, max_num_batched_tokens to
#     match. Shrinks gen time AND activation memory.
#   * ppo_micro_batch_size_per_gpu 1 -> 2 (uses the compute freed by offload-off).
#   * gpu_memory_utilization 0.6 -> 0.45 to leave room for the now-on-GPU optimizer.
#   * save_freq=5 (v3 default; was 2).
#   * NCCL_NVLS_ENABLE=0: guards against the transient "transport/nvls.cc ... Cuda
#     failure 999 'unknown error'" NVLS multicast init failure seen on base-chat
#     job 10174326 (a node/NCCL issue, NOT model-specific; it hit della-i21g1 which
#     then ran sfta00 fine). Harmless for the working runs.
#
# These verl-stock-aligned defaults (offload OFF for actor, ON for ref, gpu_mem ~0.45)
# match verl/examples/grpo_trainer/run_qwen2-7b*.sh; only the NPU examples use full
# actor offload + gpu_mem 0.3, which is where the old recipe had diverged.
#
# GPU count is configurable: NGPU=4 (default) or NGPU=2 (set --gres accordingly via
# the submit helper). Per-model knobs via env at submit time.

#SBATCH --job-name=rlm3-amplify
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- go-judge cgroup OOM fix (carried from v2; see that script for the full story) ----
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"
unset XDG_RUNTIME_DIR || true

# ---- NVLS NCCL guard (the only base-chat failure left after processor-config fix) ----
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"

# ---- amplify experiment defaults (overridable by the submit env) ----
export FS_PERTASK_REWARD_NORM="${FS_PERTASK_REWARD_NORM:-1}"
export FS_PERTASK_REWARD_NORM_TARGET="${FS_PERTASK_REWARD_NORM_TARGET:-1.0}"
export FRONTIERCS_JUDGE_MAX_WAIT="${FRONTIERCS_JUDGE_MAX_WAIT:-900}"
export FRONTIERCS_JUDGE_FAIL_SOFT="${FRONTIERCS_JUDGE_FAIL_SOFT:-1}"

export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet}"
export VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}"

# ~40 steps, save every 5 (v3 decision), no in-loop eval.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export PROJECT_NAME="${PROJECT_NAME:-rlm_amplify_v3}"
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/rlm_amplify_v3/${EXPERIMENT_NAME:-default}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rlm_amplify_v3_rollout/${EXPERIMENT_NAME:-default}}"

# ---- the SPEED config (offload off, shorter ctx, bigger micro-batch, less vLLM mem) ----
# These are read by scripts/run_verl_grpo_frontiercs_qwen35_9b.sh (now parametrized).
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"

# Shorter context (response 20k). Passed through cc_train_*'s MAX_* env overrides.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-30720}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-30720}"

export NGPU="${NGPU:-4}"
export TP="${TP:-1}"

exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@"

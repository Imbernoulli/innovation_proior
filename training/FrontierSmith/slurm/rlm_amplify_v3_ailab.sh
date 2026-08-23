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

# ---- CUDA allocator: expandable_segments cuts fragmentation. The 4-GPU smoke OOM'd on
# step 3's update_actor (tried 21GB; ~17GB free but fragmented) as responses grew toward
# 20k. expandable_segments lets the allocator grow contiguously and reclaim, which (with
# micro_batch back to 1 + gpu_mem 0.40) keeps the high-memory steps under the 141GB ceiling.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
# PARAM offload OFF (params stay on GPU -> fast fwd/bwd, the main speed win) but
# OPTIMIZER offload ON. Adam optimizer states (fp32 momentum+variance ~= 2x params,
# ~19GB/GPU for 9.4B over 4 GPUs) are only needed briefly in update_actor's optim step;
# keeping them resident (optimizer_offload=False) made GPU memory chronically high and
# the checkpoint-save gather (fsdp_checkpoint_manager.save_checkpoint -> torch.save)
# CORRECTION: smoke4b's "OOM at step-2 save" was NOT a GPU OOM — it was the CHIJ scratch
# fileset being 100% full (torch.save hit ENOSPC mid-write, leaving truncated shards). With
# both offloads OFF, step 1 ran fine at 73GB allocated (far under 141GB) and update_actor=235s
# (~3x faster than the ~700s full-offload config). So keep BOTH offloads OFF (the fast config);
# the only blocker was disk, not memory.
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
# SAFE defaults after the step-3 OOM at micro_batch=2/gpu_mem=0.45 (allocated hit 132GB
# at response mean 10k and OOM'd as responses grew toward 20k). micro_batch=1 halves the
# update_actor activation peak (the OOM site); gpu_mem 0.40 frees ~7GB of vLLM reservation.
# Offload is still OFF (the real speed win) so update_actor stays far below the old ~700s.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"

# Shorter context (response 20k). Passed through cc_train_*'s MAX_* env overrides.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-30720}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-30720}"

export NGPU="${NGPU:-4}"
export TP="${TP:-1}"

# DISK-SAFETY: rotate actor checkpoints, keeping only the latest N on disk (verl's
# checkpoint manager prunes older global_step_* dirs as new ones land). Each ckpt is
# ~36G (model-only fp32); the amplify expansion is 5 runs x ~8 saves => ~1.4TB if kept.
# Keep 2 (latest + previous, so a crash mid-save can't orphan the only ckpt) -> ~360G
# total for the expansion. Overridable; export so it reaches run_verl via env inheritance.
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"

exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@"

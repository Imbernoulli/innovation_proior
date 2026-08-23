#!/usr/bin/env bash
# rlm_amplify_v2: ONE GRPO run per starting model, on the PROVEN 4-GPU single-node
# harness (slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh -- the config that
# survived 60 full steps with checkpoints global_step_5..60). This REPLACES the
# fragile 2-GPU co-located rlm_train_2gpu_nodbus.sh chain, which kept dying ~17 min
# in from BOTH (1) a single slow FrontierCS judge call tripping the old 300s ceiling
# and (2) raylet OOM under 2-GPU co-location -- before save_freq could land a ckpt.
#
# What changed vs the fragile chain (only the launch HARNESS changes; the data/reward
# args are carried verbatim from the pending rlm_* jobs):
#   * 4 GPUs single-node + --mem=480G (proven), NOT 2-GPU co-located 750G. One full
#     node per run; 4 runs x 4 GPU = 16 GPU = the quota.
#   * proven full context: response 30000 / max_model_len 40960 (the fragile jobs
#     used the OLD truncating 16000/26624 that caused the original 0.0-reward bug).
#   * save_freq=2 so an early checkpoint lands even if a run dies; no in-loop eval.
#   * judge hardened in verl/utils/reward_score/frontiercs.py: per-problem timeout
#     900s (FRONTIERCS_JUDGE_MAX_WAIT) AND JudgeInfraError on one problem is now
#     logged + scored 0.0 (FRONTIERCS_JUDGE_FAIL_SOFT=1) instead of killing the run.
#
# Per-model knobs are passed via env at submit time (MODEL_PATH, EXPERIMENT_NAME,
# CKPT_DIR, ROLLOUT_DIR); everything else inherits the proven recipe defaults.
#
# Submit one model:
#   sbatch --job-name=rlm2_base --export=ALL,MODEL_PATH=...,EXPERIMENT_NAME=rlm2_base,\
#     CKPT_DIR=.../rlm_amplify_v2/rlm2_base,ROLLOUT_DIR=... slurm/rlm_amplify_v2_4gpu_ailab.sh
# (use scripts/rlm_amplify_v2_submit.sh to launch all 4 at once).

#SBATCH --job-name=rlm2-amplify
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- go-judge cgroup OOM fix (carried over from the fragile rlm wrapper) ----
# On this cgroup-v2 node, go-judge creates its cgroup as a systemd transient scope
# INSIDE the per-user slice (user.slice/user-372967.slice/.../gojudge-<jobid>.scope),
# which has a shared ~151 GB MemoryMax independent of the Slurm job's --mem cgroup.
# Under load the USER-SLICE OOM-killer reaps go-judge mid-run -> the node judge on
# :8092 dies -> every frontiercs reward fails. OBSERVED on rlm2_base job 10173871:
# "go-judge process exited ... current cgroup: ...user-372967.slice/.../gojudge.scope"
# then Killed. The proven 4-GPU run survived only because its node wasn't under
# user-slice memory pressure; we must NOT rely on that. Pointing
# DBUS_SESSION_BUS_ADDRESS at a dead socket makes go-judge log "connecting to systemd
# dbus failed, assuming running in container" and nest its cgroup at the unified root
# /sys/fs/cgroup instead -> escapes the user-slice cap. go-judge v1.11.1 has no CLI
# flag for this, so the dead-socket redirect is the supported lever. The shared
# scripts/start_frontiercs_judge_hybrid.sh inherits this env.
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"
unset XDG_RUNTIME_DIR || true

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"

# ---- amplify experiment defaults (overridable by the submit env) ----
# Per-task within-task reward normalization, target [0,1] -- same as the rlm_* runs.
export FS_PERTASK_REWARD_NORM="${FS_PERTASK_REWARD_NORM:-1}"
export FS_PERTASK_REWARD_NORM_TARGET="${FS_PERTASK_REWARD_NORM_TARGET:-1.0}"

# Judge hardening knobs (the code defaults already match these; set explicitly so a
# stale shell env cannot weaken them).
export FRONTIERCS_JUDGE_MAX_WAIT="${FRONTIERCS_JUDGE_MAX_WAIT:-900}"
export FRONTIERCS_JUDGE_FAIL_SOFT="${FRONTIERCS_JUDGE_FAIL_SOFT:-1}"

# The 3-task mixed parquet the proven 60-step run used (frontiercs172 + smith10 +
# alebench40) -- identical to what the pending rlm_* jobs passed.
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet}"
export VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}"

# Short, frequently-checkpointed run: ~40 steps, save EVERY 2 steps, no in-loop eval.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

# Own namespace under rlm_amplify_v2/<tag>; submit env sets the per-model values.
export PROJECT_NAME="${PROJECT_NAME:-rlm_amplify_v2}"
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/rlm_amplify_v2/${EXPERIMENT_NAME:-default}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rlm_amplify_v2_rollout/${EXPERIMENT_NAME:-default}}"

# 4-GPU single-node layout (PROVEN). Delegate to the canonical thinking-long recipe,
# which starts/health-checks the judge, sets the GRPO args and the full 30000/40960
# context, and honors MODEL_PATH / CKPT_DIR / SAVE_FREQ / TEST_FREQ / *_DATA above.
export NGPU="${NGPU:-4}"
export TP="${TP:-1}"

exec bash slurm/cc_train_qwen35_9b_mixed_thinking_long_ailab.sh "$@"

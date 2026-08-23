#!/usr/bin/env bash
# cc_rl_frontiersmith_synth.sh -- GRPO RL on the colleague's frontiersmith_synth
# problem set (500 open-ended optimization problems), with the SAME GRPO config we
# use for FrontierCS (rlm_amplify_v3). This is the "RL on our synthetic problems"
# arm to compare against "RL on FrontierCS" (rlm_amplify_v3).
#
# Difference from rlm_amplify_v3_ailab.sh: the frontiersmith_synth reward
# (verl/verl/utils/reward_score/frontiersmith_synth.py) scores OFFLINE by running
# each candidate through the synth harness (compile/generate/check under bwrap) --
# there is NO judge server. So this launcher calls run_verl_grpo_frontiercs_qwen35_9b.sh
# DIRECTLY (bypassing the frontiercs-judge health gate in
# cc_train_qwen35_9b_mixed_thinking_long_ailab.sh) and points TRAIN_DATA/VAL_DATA at
# the synth parquet. All the v3 SPEED + stability knobs are carried over verbatim.
#
# Prereqs:
#   1. .venv-vllm023 (vLLM 0.23, VERL, pandas) -- same as frontiercs
#   2. python scripts/prepare_frontiersmith_synth_parquet.py   # builds the parquet
#   3. g++ + bwrap on the compute node (both present on ailab); the synth tree at
#      $FRONTIERSMITH_SYNTH_ROOT (default ../innovation_prior/frontiersmith_synth)
#   4. GPUs (default 4; NGPU=2 also supported -- set --gres via the submit helper)
#
# Usage (direct sbatch):
#   sbatch slurm/cc_rl_frontiersmith_synth.sh
# Per-run knobs via env at submit (EXPERIMENT_NAME, MODEL_PATH, NGPU, ...).

#SBATCH --job-name=rl-fsx-synth
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"
mkdir -p logs

# ---- go-judge cgroup-OOM / dbus fix (carried from rlm_amplify_v3; harmless here) ----
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"

# CUDA_HOME fix (2026-07-22): vLLM JIT-compiles and hardcodes /usr/local/cuda-12.4/bin/nvcc,
# which some nodes lack -> "No such file or directory: .../cuda-12.4/bin/nvcc". Point CUDA_HOME
# at a CUDA that exists on every node (12.8 matches the training env); PATH gets its nvcc.
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cudatoolkit/12.8 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
[ -x "$CUDA_HOME/bin/nvcc" ] && export PATH="$CUDA_HOME/bin:$PATH"
unset XDG_RUNTIME_DIR || true

# ---- NVLS NCCL guard + CUDA allocator (same as v3) ----
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# NCCL watchdog timeout (2026-07-23): a slow research-reward evaluator stalls a rank past
# the default 600s collective timeout -> _ALLGATHER_BASE watchdog abort. Raise to 1800s and
# cap each research-reward eval (FRONTIERCS_RESEARCH_CPU_TIMEOUT) below it.
# 2026-08-01: the old 90s cap was ANTI-SIGNAL — a *correct* vdb_pareto solution needs
# 90-115s (faiss index over 1M SIFT vectors), so 90s zeroed 551 legit rollouts
# (vdb 425 / llm_sql 65 / imagenet 48). 300s: one eval <= 300s << 1800s heartbeat, and
# the per-worker MAX_CONC=1 semaphore bounds the batch tail.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export FRONTIERCS_RESEARCH_CPU_TIMEOUT="${FRONTIERCS_RESEARCH_CPU_TIMEOUT:-300}"
# 2026-08-01: research reward must use the SAME evaluator python as the standalone
# evals. Unset, the GPU harness fell back to sft_lf (no flashinfer -> qknorm died at
# import, 217 spurious zeros; no pysr either). Julia vars keep PySR's depot warm.
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_depot}"
export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_env}"
# 24 GiB RLIMIT_AS starved Julia/PySR's address-space reservation (PythonCall C.jl:63
# crashes + GC thrash: same solution 0.0 at 24G, 100.0 in ~105s at 64G). 64 GiB keeps
# the host-OOM guard (VA, not RSS; MAX_CONC=1 bounds concurrent evaluators).
export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- frontiersmith_synth reward knobs ----
# Where the colleague's synth tree lives (problems/ + harness/). The reward + the
# parquet builder both default to ../innovation_prior/frontiersmith_synth.
export FRONTIERSMITH_SYNTH_ROOT="${FRONTIERSMITH_SYNTH_ROOT:-$PROJECT_ROOT/../innovation_prior/frontiersmith_synth}"
# Fail-soft: an infra fault (missing problem, g++/bwrap trouble, harness crash) on ONE
# problem is logged + scored 0.0 rather than killing the multi-hour run (default ON),
# mirroring FRONTIERCS_JUDGE_FAIL_SOFT.
export FRONTIERSMITH_SYNTH_FAIL_SOFT="${FRONTIERSMITH_SYNTH_FAIL_SOFT:-1}"
export FRONTIERSMITH_SYNTH_MAX_WALL="${FRONTIERSMITH_SYNTH_MAX_WALL:-300}"
# The reward compiles/runs candidates on the CPU cores of this job; keep BLAS single-thread
# so many concurrent reward workers don't oversubscribe.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ---- amplify per-task reward-norm (same knobs as v3) ----
export FS_PERTASK_REWARD_NORM="${FS_PERTASK_REWARD_NORM:-1}"
export FS_PERTASK_REWARD_NORM_TARGET="${FS_PERTASK_REWARD_NORM_TARGET:-1.0}"

# ---- data: the synth parquet (built by prepare_frontiersmith_synth_parquet.py) ----
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/frontiersmith_synth/train.parquet}"
export VAL_DATA="${VAL_DATA:-$PROJECT_ROOT/data/frontiersmith_synth/full.parquet}"
# No ALE-Bench val here (keep the arm clean / comparable). Point at a non-existent
# path so run_verl falls back to VAL_DATA only.
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/__none__.parquet}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[fsx-synth] $TRAIN_DATA missing -- building it now."
    if [ -f "$PROJECT_ROOT/.venv-vllm023/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv-vllm023/bin/activate"
    fi
    python scripts/prepare_frontiersmith_synth_parquet.py \
        --synth-root "$FRONTIERSMITH_SYNTH_ROOT"
fi

# ---- training schedule (mirror rlm_amplify_v3) ----
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-60}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export PROJECT_NAME="${PROJECT_NAME:-rl_frontiersmith_synth}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fsx_synth_qwen35_9b_grpo}"
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/rl_frontiersmith_synth/${EXPERIMENT_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rl_frontiersmith_synth_rollout/${EXPERIMENT_NAME}}"

# ---- the v3 SPEED config (offload off, shorter ctx, micro_batch=1, less vLLM mem) ----
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-30720}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-30720}"

export NGPU="${NGPU:-4}"
export TP="${TP:-1}"

# Disk-safety: keep only the latest N actor checkpoints (model-only ~36G each).
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"

export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"

mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

echo "[fsx-synth] TRAIN_DATA=$TRAIN_DATA"
echo "[fsx-synth] SYNTH_ROOT=$FRONTIERSMITH_SYNTH_ROOT"
echo "[fsx-synth] MODEL=$MODEL_PATH  NGPU=$NGPU  steps=$TOTAL_TRAINING_STEPS"

# Window guard: a continuation window (FRESH_START!=1) on a run that already
# reached TOTAL_TRAINING_STEPS must be a true no-op. Without this, verl's auto
# resume loads step N, trains an off-recipe step N+1, saves it as "latest", and
# re-runs the ~3.4h final validation -- wasting 5-6h x NGPU per completed arm.
if [ "${FRESH_START:-0}" != "1" ] && [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  _done=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "$_done" -ge "$TOTAL_TRAINING_STEPS" ] 2>/dev/null; then
    echo "[window-guard] latest ckpt step=$_done >= TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS -> nothing to do, exiting."
    exit 0
  fi
fi

# Call the SAME GRPO runner FrontierCS uses; no judge server needed for synth.
# ---- bwrap/userns preflight (2026-07-30) -------------------------------------
# della disabled unprivileged user namespaces on 07-28 (user.max_user_namespaces=0).
# The frontiersmith_synth reward sandbox (harness/isorun.py = bwrap) then fails on
# EVERY candidate execution, and FRONTIERSMITH_SYNTH_FAIL_SOFT converts each failure
# into score 0.0 -- silent zero reward, no crash. The rl35b_base_sp8 run burned ~25h
# x 8 H200 producing 9600 all-zero rollouts exactly this way (reference solutions
# that scored 29.99/19.47/15.98 on 07-27 scored 0.0 on 07-30 through the same
# adapter). Abort BEFORE any GPU work when the reward path is provably dead.
# Research-data runs (frontiercs_research) use a plain subprocess reward -- exempt.
# 2026-08-01: isorun grew an apptainer backend (setuid starter, no userns needed;
# G5c leak-probe verified + reference solutions reproduce bit-exact). The preflight
# now accepts EITHER sandbox; the apptainer arm is a real end-to-end candidate
# execution via isorun self_check, not just a binary check.
if [[ "${TRAIN_DATA:-}" == *frontiersmith_synth* && "${SKIP_BWRAP_PREFLIGHT:-0}" != "1" ]]; then
  _ISORUN="${FRONTIERSMITH_SYNTH_ROOT:-/scratch/gpfs/CHIJ/bohan/fs/innovation_prior/frontiersmith_synth}/harness/isorun.py"
  if bwrap --ro-bind / / true 2>/dev/null; then
    echo "[preflight] sandbox backend: bwrap"
  elif ISORUN_BACKEND=apptainer python3 "$_ISORUN" 2>/dev/null | grep -q "sandbox=apptainer"; then
    echo "[preflight] bwrap dead; apptainer sandbox verified via real isorun self_check"
    export ISORUN_BACKEND=apptainer   # pin so reward workers skip the dead-bwrap probe
  else
    echo "FATAL: no candidate sandbox on $(hostname): bwrap AND apptainer both dead." >&2
    echo "       synth reward would FAIL_SOFT to 0.0 for EVERY rollout (silent no-op training)." >&2
    echo "       Bypass (not advised): SKIP_BWRAP_PREFLIGHT=1" >&2
    exit 1
  fi
fi

# 2026-08-04: the July-proven sampling lived ONLY in the submit wrapper, so running
# this launcher directly inherited stock verl (top_p 1.0 / top_k -1) -> generations
# never stop -> 83% hit the token cap, 17% close </think>, rewards all zero.
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
export ROLLOUT_PRESENCE_PENALTY="${ROLLOUT_PRESENCE_PENALTY:-1.5}"

exec bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh "$@"

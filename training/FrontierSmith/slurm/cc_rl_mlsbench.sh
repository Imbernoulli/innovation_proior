#!/usr/bin/env bash
# cc_rl_mlsbench.sh -- GRPO RL on MLS-Bench agentic episodes (multi-turn tool use
# inside Apptainer task environments), via the MLSBenchAgentLoop in our verl fork.
#
# Architecture (see experiments/MLS_RL_INTEGRATION_zh.md):
#   * every rollout sample runs a FULL MLS-Bench episode (edit/test/submit tools,
#     Apptainer execution) driven token-in-token-out by
#     verl/verl/experimental/agent_loop/mlsbench_agent_loop.py
#   * reward = deterministic `mlsbench score` task score of the episode's
#     submission (computed in-loop, lands in rm_scores; fail-soft -> 0.0)
#   * ADDITIVE + env-gated: activation requires BOTH the agent_loop_config_path
#     hydra override below AND parquet rows with agent_name=mlsbench_agent.
#     Nothing in the single-turn synth/FrontierCS path is touched.
#
# SMOKE defaults: 4 dev tasks x n=2 rollouts, 2 GPUs, 3 steps, save_freq=2.
# Scale-up knobs documented in the MD doc.
#
# Usage:
#   sbatch slurm/cc_rl_mlsbench.sh
#   (or scripts/cc_rl_mlsbench_submit.sh for GPU/CPU/mem-scaled submission)
#
#SBATCH --job-name=rl-mlsbench
#SBATCH --partition=ailab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"
mkdir -p logs

# ---- guards carried from the proven synth recipe ----
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"
unset XDG_RUNTIME_DIR || true
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ---- MLS-Bench RL episode env (read by mlsbench_agent_loop.py in Ray workers) ----
export MLS_RL_MLSBENCH_ROOT="${MLS_RL_MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}"
export MLS_RL_DATA_ROOT="${MLS_RL_DATA_ROOT:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
# Episode environment worker: one subprocess per episode (task-module isolation +
# clean kill), run under the PROVEN eval conda python (has mlsbench deps incl pgmpy).
export MLS_RL_WORKER_PYTHON="${MLS_RL_WORKER_PYTHON:-/home/bl3615/miniconda3/bin/python}"
export MLS_RL_WORKER_SCRIPT="${MLS_RL_WORKER_SCRIPT:-$PROJECT_ROOT/scripts/mlsbench_rl_episode_worker.py}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mlsrl_qwen35_9b_grpo_smoke}"
export MLS_RL_OUTPUT_BASE="${MLS_RL_OUTPUT_BASE:-$PROJECT_ROOT/outputs/mls_rl/${EXPERIMENT_NAME}}"
export MLS_RL_MAX_STEPS="${MLS_RL_MAX_STEPS:-8}"       # DEFAULT action budget (parquet extra_info.budget overrides per row)
export MLS_RL_MAX_TESTS="${MLS_RL_MAX_TESTS:-1}"       # DEFAULT test() budget  (two-tier parquet: tight 5/1, loose 20/3)
export MLS_RL_USE_REPLACE="${MLS_RL_USE_REPLACE:-1}"   # str_replace edit schema (devfix口径)
export MLS_RL_MAX_CONC_TESTS="${MLS_RL_MAX_CONC_TESTS:-3}"   # concurrent Apptainer tests / worker
export MLS_RL_EPISODE_TIMEOUT="${MLS_RL_EPISODE_TIMEOUT:-3000}"
export MLS_RL_NUDGE_LIMIT="${MLS_RL_NUDGE_LIMIT:-2}"
export MLS_RL_KEEP_WORKSPACE="${MLS_RL_KEEP_WORKSPACE:-0}"
export MLSBENCH_SCHEDULER_MANAGED=1
export MLSBENCH_NO_PREBUILT=1
mkdir -p "$MLS_RL_OUTPUT_BASE"

# ---- data: MLS-Bench dev-task parquet ----
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/mlsbench_rl/train_smoke.parquet}"
export VAL_DATA="${VAL_DATA:-$TRAIN_DATA}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/__none__.parquet}"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "[mls-rl] ERROR: $TRAIN_DATA missing. Build it with:" >&2
    echo "  python scripts/prepare_mlsbench_rl_parquet.py [--tasks ...]" >&2
    exit 1
fi

# ---- SMOKE training shape (episodes are minutes-long; keep the batch tiny) ----
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-60}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export PROJECT_NAME="${PROJECT_NAME:-rl_mlsbench}"
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/rl_mlsbench/${EXPERIMENT_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rl_mlsbench_rollout/${EXPERIMENT_NAME}}"

# ---- GRPO shape: 4 prompts x 2 rollouts = 8 concurrent episodes ----
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export VAL_N="${VAL_N:-1}"

# ---- memory/speed knobs (2-GPU-safe: optimizer offload ON like smoke2c) ----
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"

# ---- context: initial prompts are 3k-11k tokens for the smoke tasks ----
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-14336}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-14336}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-30720}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-30720}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

export NGPU="${NGPU:-2}"
export TP="${TP:-1}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"

mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

echo "[mls-rl] TRAIN_DATA=$TRAIN_DATA"
echo "[mls-rl] MLSBENCH_ROOT=$MLS_RL_MLSBENCH_ROOT"
echo "[mls-rl] MODEL=$MODEL_PATH NGPU=$NGPU steps=$TOTAL_TRAINING_STEPS batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N}"
echo "[mls-rl] episode: max_steps=$MLS_RL_MAX_STEPS max_tests=$MLS_RL_MAX_TESTS timeout=${MLS_RL_EPISODE_TIMEOUT}s conc_tests=$MLS_RL_MAX_CONC_TESTS"
echo "[mls-rl] outputs: $MLS_RL_OUTPUT_BASE (episode_logs/ has per-episode JSONL)"

# window guard (same as synth launcher)
if [ "${FRESH_START:-0}" != "1" ] && [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  _done=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "$_done" -ge "$TOTAL_TRAINING_STEPS" ] 2>/dev/null; then
    echo "[window-guard] latest ckpt step=$_done >= $TOTAL_TRAINING_STEPS -> nothing to do."
    exit 0
  fi
fi

# ---- run the proven GRPO runner + the MLS agent-loop overrides ----
# agent_loop_config_path activates MLSBenchAgentLoop ONLY for this run;
# num_workers=2 keeps the per-node Apptainer fan-out bounded (2 workers x
# MLS_RL_MAX_CONC_TESTS concurrent tests).
exec bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/config/mlsbench_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers="${MLS_RL_AGENT_WORKERS:-2}" \
    "$@"

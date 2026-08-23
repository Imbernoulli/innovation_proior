#!/usr/bin/env bash
# NO-TRAIN harness test for the MLS-Bench RL agent loop.
# Starts a local vLLM server on this job's GPU, then drives 1-2 REAL MLS-Bench
# episodes through MLSBenchAgentLoop (workspace + Apptainer test + XML tool parse
# + mask building + mlsbench scoring), printing per-turn token counts.
#
# Usage:
#   sbatch --export=ALL slurm/cc_mlsbench_rl_episode_test.sh
#   env knobs: MODEL_PATH, TASKS ("t1 t2"), EPISODES, MLS_RL_MAX_STEPS, MLS_RL_MAX_TESTS
#   per-episode budget tiers (extra_info.budget path, same as training rows):
#     BUDGET_TIER=tight|loose   [PARQUET=data/mlsbench_rl/train_tv1.parquet]
#
#SBATCH --job-name=cc-mlsrl-eptest
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"
mkdir -p logs

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
TASKS="${TASKS:-ml-clustering-algorithm}"
EPISODES="${EPISODES:-1}"
BUDGET_TIER="${BUDGET_TIER:-}"   # tight|loose -> per-episode extra_info.budget
PARQUET="${PARQUET:-}"           # optional: replay the real parquet row's extra_info

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# ---- MLS-Bench RL env (same knobs the training launcher sets) ----
export MLS_RL_MLSBENCH_ROOT="${MLS_RL_MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}"
export MLS_RL_DATA_ROOT="${MLS_RL_DATA_ROOT:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
export MLS_RL_WORKER_PYTHON="${MLS_RL_WORKER_PYTHON:-/home/bl3615/miniconda3/bin/python}"
export MLS_RL_WORKER_SCRIPT="${MLS_RL_WORKER_SCRIPT:-$PROJECT_ROOT/scripts/mlsbench_rl_episode_worker.py}"
export MLS_RL_OUTPUT_BASE="${MLS_RL_OUTPUT_BASE:-$PROJECT_ROOT/outputs/mls_rl_episode_test}"
export MLS_RL_MAX_STEPS="${MLS_RL_MAX_STEPS:-8}"
export MLS_RL_MAX_TESTS="${MLS_RL_MAX_TESTS:-1}"
export MLS_RL_EPISODE_TIMEOUT="${MLS_RL_EPISODE_TIMEOUT:-5400}"
export MLS_RL_KEEP_WORKSPACE="${MLS_RL_KEEP_WORKSPACE:-1}"   # keep for inspection in the test
export MLSBENCH_SCHEDULER_MANAGED=1
export MLSBENCH_NO_PREBUILT=1
mkdir -p "$MLS_RL_OUTPUT_BASE"

# ---- local vLLM server (raw /v1/completions; no tool parser needed) ----
JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
PORT=$(( 35000 + JOBU ))
TAG="mlsrl-eptest"
PORT="$PORT" SERVED_MODEL_NAME="$TAG" MODEL_PATH="$MODEL_PATH" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" GPU_MEMORY_UTILIZATION=0.85 \
  scripts/start_vllm_server.sh &
VLLM_PID="$!"
cleanup() { kill "$VLLM_PID" >/dev/null 2>&1 || true; wait "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[eptest] waiting for vLLM on :$PORT ..."
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 2
  kill -0 "$VLLM_PID" >/dev/null 2>&1 || { echo "vLLM exited early" >&2; exit 1; }
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "no /v1/models" >&2; exit 1; }
echo "[eptest] vLLM ready."

source "$PROJECT_ROOT/.venv-vllm023/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_ARGS=()
[ -n "$BUDGET_TIER" ] && EXTRA_ARGS+=(--budget-tier "$BUDGET_TIER")
[ -n "$PARQUET" ] && EXTRA_ARGS+=(--parquet "$PARQUET")

# shellcheck disable=SC2086
python scripts/mlsbench_rl_episode_test.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$TAG" \
  --model-path "$MODEL_PATH" \
  --tasks $TASKS \
  --episodes "$EPISODES" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[eptest] DONE. episode logs: $MLS_RL_OUTPUT_BASE/episode_logs/"

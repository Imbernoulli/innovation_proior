#!/usr/bin/env bash
# Sharded trained-checkpoint eval for Qwen3.5-9B mixed public GRPO.
# Runs FrontierCS + ALE-Bench through vLLM with Qwen3.5 thinking-general params.
#
# Submit:
#   sbatch slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh

#SBATCH --job-name=fs-q35mix-think
#SBATCH --partition=ailab
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-${SHARD_IDX:-0}}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm}"
PORT_OFFSET=$((SHARD_IDX * 10))

export SOURCE="${SOURCE:-both}"
export OUTPUT_DIR="${OUTPUT_BASE}/shards/shard_${SHARD_IDX}"
export SAMPLES_JSONL="${OUTPUT_DIR}/samples.jsonl"
export SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
export NUM_SHARDS="$NUM_SHARDS"
export SHARD_IDX="$SHARD_IDX"
export N_SAMPLES="${N_SAMPLES:-5}"

export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/cc_qwen35_9b_mixed_hf_fixed}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-cc-qwen35-9b-mixed-fixed-think-${SHARD_IDX}}"
export TP="${TP:-1}"
if [ -z "${MAX_MODEL_LEN:-}" ]; then
  MAX_MODEL_LEN="$(
    MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os
from pathlib import Path

cfg_path = Path(os.environ["MODEL_PATH"]) / "config.json"
default = 49152
try:
    cfg = json.loads(cfg_path.read_text())
    max_pos = cfg.get("max_position_embeddings")
    if isinstance(max_pos, int) and max_pos > 0:
        print(min(default, max_pos))
    else:
        print(default)
except Exception:
    print(default)
PY
  )"
fi
export MAX_MODEL_LEN
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0.0}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export CONCURRENCY="${CONCURRENCY:-64}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-2400}"

export FRONTIERCS_PROMPT_SOURCE="${FRONTIERCS_PROMPT_SOURCE:-official}"
export FRONTIERCS_SCORE_BACKEND="${FRONTIERCS_SCORE_BACKEND:-official}"
export SAVE_TEXT="${SAVE_TEXT:-0}"
export TEXT_PREVIEW_CHARS="${TEXT_PREVIEW_CHARS:-0}"
export SEED="${SEED:-42}"

export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
# Job-unique ports + cgroup so co-located eval jobs on one node never collide
# (go-judge's default fixed "gojudge.scope" cgroup unit and fixed ports clash otherwise).
JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
export PORT="${PORT:-$(( 20000 + JOBU + SHARD_IDX ))}"
export VLLM_PORT="${VLLM_PORT:-$(( 33000 + JOBU + SHARD_IDX ))}"
export GJ_PORT="${GJ_PORT:-$(( 46000 + JOBU + SHARD_IDX ))}"
export GJ_CGROUP_PREFIX="${GJ_CGROUP_PREFIX:-gojudge-${SLURM_JOB_ID:-$$}-${SHARD_IDX}}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-mixed-eval-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}-${SHARD_IDX}}"

exec bash slurm/eval_qwen35_9b_base_vllm_ailab.sh "$@"

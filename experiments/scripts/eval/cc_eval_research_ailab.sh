#!/usr/bin/env bash
# =============================================================================
# Frontier-CS RESEARCH-track eval (thinking mode), single 1-GPU job.
#
# Unlike the algorithmic eval (cc_eval_thinking_both_ailab.sh) the research track
# does NOT use the C++ go-judge. Instead each problem is scored by running the
# official research evaluator.py DIRECTLY on this node's GPU (Docker is not
# available on this cluster) via scripts/frontiercs_research_eval.py. So:
#   1. Serve the model via vLLM (THINKING mode, 32K budget) -- but at a REDUCED
#      gpu-memory-utilization so the Triton-kernel evaluators have GPU headroom
#      to compile+run their kernels on the SAME GPU.
#   2. Generate research solutions (Python `Solution` class) and score them with
#      the official evaluator (score 0..100, leaderboard scale). Infra failures
#      (import error / missing GPU / timeout) RAISE and are recorded as `error`,
#      never silently scored 0.
#   3. Write summary.json with the frontiercs_research metrics (Avg@5 / best@5).
#
# Usage:
#   sbatch --job-name=cc-research-<TAG> \
#     --export=ALL,MODEL_PATH=<MODEL_DIR>,TAG=<TAG> \
#     slurm/cc_eval_research_ailab.sh
#
# By default scores the GPU research subset (RESEARCH_DATA points to the GPU
# parquet). Set RESEARCH_DATA=.../research.parquet for all 64 runnable problems
# (CPU problems may need extra per-problem deps; see frontiercs_research_eval.py).
# Short validation: VALIDATE=1 (RESEARCH_LIMIT small).
# =============================================================================
#SBATCH --job-name=cc-eval-research
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
cd "$PROJECT_ROOT"
mkdir -p logs

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-9B-bf16}}"
TAG="${TAG:-$(basename "$MODEL_PATH")}"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "ERROR: MODEL_PATH=$MODEL_PATH has no config.json" >&2
  exit 1
fi
export MODEL_PATH

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_${TAG}_research_thinking_32k_vllm}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/shard_0}"
export SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_BASE/summary.json}"
export SAMPLES_JSONL="${SAMPLES_JSONL:-$OUTPUT_DIR/samples.jsonl}"
mkdir -p "$OUTPUT_BASE" "$OUTPUT_DIR"

# Which research problems. Default: GPU subset (self-contained Triton problems).
export RESEARCH_DATA="${RESEARCH_DATA:-$PROJECT_ROOT/data/frontiercs/research_gpu.parquet}"

# Interpreter that RUNS the research evaluators (needs torch+triton 3.6 for the
# Triton-kernel benchmark harness; see frontiercs_research_eval._default_research_python).
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf/bin/python}"
export FRONTIERCS_RESEARCH_TIMEOUT="${FRONTIERCS_RESEARCH_TIMEOUT:-1200}"

exec > >(grep -avE 'real_accelerator|ds_accelerator|Setting ds_accelerator') \
    2> >(grep -avE 'real_accelerator|ds_accelerator|Setting ds_accelerator' >&2)

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TMPDIR="/tmp"

JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
export VLLM_PORT="${VLLM_PORT:-$(( 34000 + JOBU ))}"

echo "[cc_research] TAG=$TAG MODEL=$MODEL_PATH"
echo "[cc_research] RESEARCH_DATA=$RESEARCH_DATA python=$FRONTIERCS_RESEARCH_PYTHON"
echo "[cc_research] vLLM=$VLLM_PORT output=$SUMMARY_JSON"

# ----- thinking budget (same as algorithmic eval) -----------------------------
export TP="${TP:-1}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
if [ -z "${MAX_MODEL_LEN:-}" ]; then
  MAX_MODEL_LEN="$(
    MODEL_PATH="$MODEL_PATH" MAX_TOKENS="$MAX_TOKENS" python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
cap = cfg.get("max_position_embeddings") or 262144
mt = int(os.environ["MAX_TOKENS"])
want = 9000 + mt
print(min(cap, want))
PY
  )"
fi
export MAX_MODEL_LEN
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# CRITICAL: leave GPU headroom for the Triton evaluators that run on this SAME
# GPU. 0.55 on a 9B model keeps ~enough free VRAM for kernel compile+bench.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-cc-${TAG}-research}"
export HOST=127.0.0.1

# ----- thinking-mode sampling -------------------------------------------------
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0.0}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
# Lower concurrency: each research score is a GPU subprocess; avoid GPU thrash.
export CONCURRENCY="${CONCURRENCY:-4}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-2400}"

export N_SAMPLES="${N_SAMPLES:-5}"
export SEED="${SEED:-42}"
export SAVE_TEXT="${SAVE_TEXT:-1}"
# Research evaluators can legitimately time out / hit transient issues on a few
# heavy problems; tolerate a handful (recorded loudly as error) rather than
# hard-failing the whole job before summary.json is written.
export MAX_ERRORS="${MAX_ERRORS:-20}"

if [ "${VALIDATE:-0}" = "1" ]; then
  export RESEARCH_LIMIT="${RESEARCH_LIMIT:-3}"
  echo "[cc_research] VALIDATE mode: RESEARCH_LIMIT=$RESEARCH_LIMIT"
fi

# ----- start vLLM -------------------------------------------------------------
PORT="$VLLM_PORT" scripts/start_vllm_server.sh &
VLLM_PID="$!"
cleanup() { kill "$VLLM_PID" >/dev/null 2>&1 || true; wait "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[cc_research] waiting for vLLM /health ..."
for _ in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then echo "vLLM server exited early" >&2; exit 1; fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
  || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
echo "[cc_research] vLLM ready on ${VLLM_PORT}"

# ----- run eval (research source; no go-judge needed) -------------------------
PROJECT_ROOT_LOCAL="$PROJECT_ROOT"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "ERROR: .venv not found" >&2; exit 1
fi
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src:$PROJECT_ROOT/scripts:$PROJECT_ROOT/.cache/Frontier-CS-official:$PROJECT_ROOT/.cache/Frontier-CS-official/src${PYTHONPATH:+:$PYTHONPATH}"

RESUME_FLAG=()
[ "${RESUME:-1}" = "1" ] && RESUME_FLAG=(--resume)
THINK_FLAG=(--enable-thinking)
[ "${ENABLE_THINKING:-1}" = "0" ] && THINK_FLAG=(--no-enable-thinking)
LIMIT_FLAG=()
[ -n "${RESEARCH_LIMIT:-}" ] && LIMIT_FLAG=(--limit-research "$RESEARCH_LIMIT")

python scripts/eval_qwen35_base_vllm_request.py \
  --source research \
  --research-data "$RESEARCH_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --samples-jsonl "$SAMPLES_JSONL" \
  --summary-json "$SUMMARY_JSON" \
  --n-samples "$N_SAMPLES" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --model "$SERVED_MODEL_NAME" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" \
  --presence-penalty "$PRESENCE_PENALTY" --repetition-penalty "$REPETITION_PENALTY" \
  --concurrency "$CONCURRENCY" --timeout "$REQUEST_TIMEOUT" \
  --seed "$SEED" --max-errors "$MAX_ERRORS" --save-text \
  "${THINK_FLAG[@]}" "${RESUME_FLAG[@]}" "${LIMIT_FLAG[@]}"

echo "[cc_research] DONE. summary -> $SUMMARY_JSON"

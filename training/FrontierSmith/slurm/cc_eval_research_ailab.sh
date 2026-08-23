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
# By default scores the FULL 64-problem runnable leaderboard scope (21 GPU + 43
# CPU); this is the official research set minus the 4 poc_generation categories,
# which require Docker-in-Docker and cannot run here. Narrow with
# RESEARCH_SCOPE=gpu (21) or RESEARCH_SCOPE=cpu (43), or point RESEARCH_DATA at an
# explicit parquet. CPU problems may need extra per-problem deps (research_overlay
# env); see frontiercs_research_eval.py. Short validation: VALIDATE=1.
# =============================================================================
#SBATCH --job-name=cc-eval-research
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
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

# Which research problems.
#   RESEARCH_SCOPE=full  (DEFAULT, FAITHFUL) -> all 64 runnable leaderboard problems
#                                               (21 GPU + 43 CPU). This is the
#                                               official scope minus the 4
#                                               poc_generation categories, which
#                                               need Docker-in-Docker (impossible
#                                               here). See frontiercs_research_eval.py.
#   RESEARCH_SCOPE=gpu    -> 21 self-contained Triton GPU problems only (fast).
#   RESEARCH_SCOPE=cpu    -> 43 CPU-only problems.
# RESEARCH_DATA (an explicit parquet path) overrides RESEARCH_SCOPE.
RESEARCH_SCOPE="${RESEARCH_SCOPE:-full}"
if [ -z "${RESEARCH_DATA:-}" ]; then
  case "$RESEARCH_SCOPE" in
    gpu)  RESEARCH_DATA="$PROJECT_ROOT/data/frontiercs/research_gpu.parquet" ;;
    cpu)  RESEARCH_DATA="$PROJECT_ROOT/data/frontiercs/research_cpu.parquet" ;;
    full|all|*) RESEARCH_DATA="$PROJECT_ROOT/data/frontiercs/research.parquet" ;;
  esac
fi
export RESEARCH_DATA

# Interpreter that RUNS the research evaluators. Use envs/research_overlay: a
# --system-site-packages venv on top of envs/sft_lf, so the GPU Triton harness
# still gets torch 2.10 + triton 3.6 from the base, PLUS the CPU-subset deps
# (faiss-cpu, pysr/juliacall, coverage, sky_spot, openai, ...) the 43 CPU
# research problems need. Falls back to sft_lf if the overlay is absent.
_OVL=/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python
[ -x "$_OVL" ] || _OVL=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf/bin/python
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-$_OVL}"
export FRONTIERCS_RESEARCH_TIMEOUT="${FRONTIERCS_RESEARCH_TIMEOUT:-1200}"
# 24 GiB VA is thin for CUDA-context+tensors on H200 (torch+triton import alone maps
# 13.4 GiB VmPeak); the dedicated eval node has RAM, so give evaluators 48 GiB here.
# RL keeps 24 (there the cap IS the host-RAM guard).
export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-48}"
# CPU-subset evaluators get a longer wall (imagenet trains a CNN on CPU; vdb
# builds a 1M-vector index; symbolic_regression JITs SymbolicRegression.jl).
export FRONTIERCS_RESEARCH_CPU_TIMEOUT="${FRONTIERCS_RESEARCH_CPU_TIMEOUT:-2400}"
# Julia depot/project provisioned offline for symbolic_regression (pysr).
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_depot}"
export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_env}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"

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

# Port assignment: vLLM binds BOTH the API port AND an internal EngineCore
# TCPStore port at API_port+1. So co-located jobs whose API ports are only 1
# apart (adjacent SLURM job IDs -> adjacent JOBU) COLLIDE: job A's EngineCore
# (api+1) lands on job B's API port. Earlier failures (10189807/8/11) were all
# EADDRINUSE from exactly this. Fix: stride the port by 8 per job so each job
# owns an 8-wide block (api .. api+7) with no overlap, and add a small random
# jitter block so a re-run that happens to reuse a JOBU avoids a lingering
# TIME_WAIT socket. Range stays within an unprivileged, rarely-used band.
JOBU=$(( ${SLURM_JOB_ID:-$$} % 4000 ))
JITTER=$(( (RANDOM % 13) * 8 ))          # 0,8,...,96
export VLLM_PORT="${VLLM_PORT:-$(( 30000 + JOBU * 8 + JITTER ))}"

echo "[cc_research] TAG=$TAG MODEL=$MODEL_PATH"
echo "[cc_research] RESEARCH_DATA=$RESEARCH_DATA python=$FRONTIERCS_RESEARCH_PYTHON"
echo "[cc_research] vLLM=$VLLM_PORT (block ${VLLM_PORT}..$((VLLM_PORT+7))) output=$SUMMARY_JSON"

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
export FRONTIERCS_RESEARCH_PROMPT_STYLE="${FRONTIERCS_RESEARCH_PROMPT_STYLE:-concat}"  # match official single-string system+user prompt (was defaulting to separate chat roles)
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
# GPU keep-alive: a tiny CUDA matmul every 30s so the long CPU-bound scoring
# phase (official evaluators run on CPU, GPU idle) does not trip the cluster's
# 90-min 0%-util auto-cancel. ~300MB CUDA context, negligible vs vLLM's share.
"$_OVL" -c "import torch,time
while True:
    (torch.randn(256,256,device='cuda')@torch.randn(256,256,device='cuda')).sum().item(); time.sleep(30)" >/dev/null 2>&1 &
KEEPALIVE_PID="$!"
cleanup() { kill "$VLLM_PID" "${KEEPALIVE_PID:-}" >/dev/null 2>&1 || true; wait "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[cc_research] waiting for vLLM /health ..."
for _ in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then echo "vLLM server exited early" >&2; exit 1; fi
done
# /health (and even /v1/models returning 200) can come up BEFORE the model is
# actually registered, especially for slow-loading checkpoints (e.g. the 36G
# methodtraj build). The OLD check only verified /v1/models responded, so the eval
# could fire into an empty server and 404 every sample in <1s (jobs 10199300/01).
# Harden: poll until the SERVED model name appears in /v1/models, then sanity-check
# one completion actually returns (not 404), before starting the eval.
echo "[cc_research] waiting for served model '${SERVED_MODEL_NAME}' to register ..."
_registered=0
for _ in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | grep -qF "\"${SERVED_MODEL_NAME}\""; then _registered=1; break; fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then echo "vLLM server exited early during load" >&2; exit 1; fi
done
[ "$_registered" = 1 ] || { echo "ERROR: served model '${SERVED_MODEL_NAME}' never registered in /v1/models" >&2; exit 1; }
# Final guard: a real completion must not 404 (catches any residual race).
for _ in $(seq 1 30); do
  _code=$(curl -fsS -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"ping\",\"max_tokens\":1}" 2>/dev/null || echo 000)
  [ "$_code" = 200 ] && break
  sleep 2
done
[ "${_code:-000}" = 200 ] || { echo "ERROR: served model '${SERVED_MODEL_NAME}' not answering completions (last http=$_code)" >&2; exit 1; }
echo "[cc_research] vLLM ready on ${VLLM_PORT} (model registered + completion 200)"

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
  --num-shards "${NUM_SHARDS:-1}" \
  --shard-idx "${SHARD_IDX:-0}" \
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

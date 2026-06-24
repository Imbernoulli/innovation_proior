#!/usr/bin/env bash
# =============================================================================
# Reusable MLS-Bench CPU-task eval for a vLLM-served model — ONE self-contained
# Slurm job, NO nested Slurm.
#
# Design (mirrors slurm/cc_eval_thinking_both_ailab.sh):
#   1. Serve the model via a local vLLM OpenAI-compatible server on the job's
#      single GPU; health-gate on /health + /v1/models.
#   2. Run the MLS-Bench CPU tasks with `mlsbench agent`, using a generated
#      config that has container_runtime=apptainer and NO `slurm:` block, so each
#      task's Apptainer container runs LOCALLY (direct subprocess) on this node —
#      no sbatch/srun, no nested Slurm. The CPU-only task containers share the
#      node's CPUs while vLLM holds the GPU (no GPU contention).
#   3. A bounded in-job worker pool (CONCURRENCY) runs the tasks; each has a hard
#      per-task timeout, so a slow/hanging task never blocks the rest.
#   4. Score every task via `mlsbench score`, collect into summary.json.
#
# Eval口径: EVAL_RESEARCHER_YEAR=2026 (researcher system prompt prepended inside
# MLS-Bench), THINKING mode ON (Qwen3 enable_thinking via chat_template_kwargs).
#
# Usage:
#   sbatch --job-name=cc-mlsbench-cpu-<TAG> \
#     --export=ALL,MODEL_PATH=<MODEL_DIR>,TAG=<TAG> \
#     slurm/cc_eval_mlsbench_cpu_ailab.sh
#
#   SMOKE (1 task): add SMOKE_TASK=ml-clustering-algorithm (or TASKS="a b c"),
#   and optionally LIMIT=1.
# =============================================================================
#SBATCH --job-name=cc-mlsbench-cpu
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

# ----- inputs -----------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3-8B}}"
TAG="${TAG:-$(basename "$MODEL_PATH")}"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "ERROR: MODEL_PATH=$MODEL_PATH has no config.json" >&2
  exit 1
fi
export MODEL_PATH

MLSBENCH_ROOT="${MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench}"
if [ ! -d "$MLSBENCH_ROOT/src/mlsbench" ]; then
  echo "ERROR: MLSBENCH_ROOT=$MLSBENCH_ROOT is not an MLS-Bench checkout" >&2
  exit 1
fi

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_mlsbench_cpu_${TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_BASE/summary.json}"
mkdir -p "$OUTPUT_BASE"

# ----- env --------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/tmp}"

# Eval口径: researcher system prompt (year 2026) + thinking on.
export EVAL_RESEARCHER_YEAR="${EVAL_RESEARCHER_YEAR:-2026}"

# Offline / no-nested-Slurm guards for MLS-Bench.
export MLSBENCH_NO_PREBUILT="${MLSBENCH_NO_PREBUILT:-1}"   # never pull SIFs (no internet on compute)
export MLSBENCH_SCHEDULER_MANAGED=1                        # force inline apptainer (no daemon)

# ----- job-unique vLLM port ---------------------------------------------------
JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
export VLLM_PORT="${VLLM_PORT:-$(( 34000 + JOBU ))}"
export HOST=127.0.0.1
SERVED_MODEL_NAME="vllm/${TAG}"      # full string sent to vLLM; must match --model

# ----- thinking-mode vLLM config (Qwen3 thinking-general口径) ------------------
export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export DTYPE="${DTYPE:-bfloat16}"

# ----- CPU-task worker pool config --------------------------------------------
export CONCURRENCY="${CONCURRENCY:-20}"   # run all 20 CPU tasks in parallel (one wave), bounded by the slowest task
export TASK_TIMEOUT="${TASK_TIMEOUT:-5400}"   # 90 min per task

echo "[mlsbench-cpu] TAG=$TAG MODEL=$MODEL_PATH"
echo "[mlsbench-cpu] MLSBENCH_ROOT=$MLSBENCH_ROOT vLLM_PORT=$VLLM_PORT served=$SERVED_MODEL_NAME"
echo "[mlsbench-cpu] EVAL_RESEARCHER_YEAR=$EVAL_RESEARCHER_YEAR CONCURRENCY=$CONCURRENCY TASK_TIMEOUT=${TASK_TIMEOUT}s"
echo "[mlsbench-cpu] summary -> $SUMMARY_JSON"

# ----- generate the MLS-Bench config (apptainer + NO slurm block) -------------
# data_root points at the shared, prepared MLS-Bench data tree. The vllm provider
# points at our local server with a dummy key. A bare base_url ending in /v1 is
# used verbatim by MLS-Bench's OpenAI client.
DATA_ROOT="${MLSBENCH_DATA_ROOT:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
SAVE_PATH="${MLSBENCH_SAVE_PATH:-$OUTPUT_BASE/saves}"
mkdir -p "$SAVE_PATH"
GEN_CONFIG="$OUTPUT_BASE/config_vllm_local_${SLURM_JOB_ID:-manual}.yaml"
cat > "$GEN_CONFIG" <<YAML
# AUTO-GENERATED by cc_eval_mlsbench_cpu_ailab.sh — local Apptainer, NO slurm block.
max_steps: ${MLSBENCH_MAX_STEPS:-20}
max_tests: ${MLSBENCH_MAX_TESTS:-3}
save_path: ${SAVE_PATH}
data_root: ${DATA_ROOT}
seeds: [42]
container_runtime: apptainer

thinking:
  enabled: true
  reasoning_effort: "high"
  budget_tokens: ${MLSBENCH_BUDGET_TOKENS:-10000}

providers:
  vllm:
    api_key: "EMPTY"
    base_url: "http://127.0.0.1:${VLLM_PORT}/v1"
YAML
echo "[mlsbench-cpu] generated config: $GEN_CONFIG"
cat "$GEN_CONFIG" | sed 's/^/    /'

# ----- start vLLM -------------------------------------------------------------
# MLS-Bench's InteractiveAgent uses OpenAI tool-calling with tool_choice="required".
# vLLM rejects that with HTTP 400 unless a tool-call parser is enabled, so we pass
# --enable-auto-tool-choice --tool-call-parser hermes (Qwen3 uses hermes-style tool
# calls). These extra args are forwarded by start_vllm_server.sh's `exec vllm serve
# ... "$@"`. Kept here (not in start_vllm_server.sh) so FrontierCS/ALE/Theta evals,
# which don't use tools, are unaffected.
PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  scripts/start_vllm_server.sh --enable-auto-tool-choice --tool-call-parser hermes &
VLLM_PID="$!"

cleanup() { kill "$VLLM_PID" >/dev/null 2>&1 || true; wait "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[mlsbench-cpu] waiting for vLLM /health on ${VLLM_PORT} ..."
for _ in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then echo "vLLM exited early" >&2; exit 1; fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
  || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
echo "[mlsbench-cpu] vLLM ready on ${VLLM_PORT} (served=$SERVED_MODEL_NAME)"

# ----- run the CPU-task worker pool (local Apptainer, no nested Slurm) ---------
# Use the conda python that has mlsbench + openai installed (the `mlsbench`
# console script lives there). Fall back to whatever python3 is on PATH.
MLSBENCH_PY="${MLSBENCH_PY:-/home/bl3615/miniconda3/bin/python}"
[ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"

EXTRA_ARGS=()
if [ -n "${SMOKE_TASK:-}" ]; then EXTRA_ARGS+=(--tasks "$SMOKE_TASK"); fi
if [ -n "${TASKS:-}" ]; then EXTRA_ARGS+=(--tasks $TASKS); fi
if [ -n "${LIMIT:-}" ]; then EXTRA_ARGS+=(--limit "$LIMIT"); fi

echo "[mlsbench-cpu] launching worker pool with $MLSBENCH_PY"
MODEL="$SERVED_MODEL_NAME" \
MLSBENCH_ROOT="$MLSBENCH_ROOT" \
"$MLSBENCH_PY" "$PROJECT_ROOT/scripts/mlsbench_run_cpu_tasks.py" \
  --config "$GEN_CONFIG" \
  --model "$SERVED_MODEL_NAME" \
  --root "$MLSBENCH_ROOT" \
  --out "$SUMMARY_JSON" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TASK_TIMEOUT" \
  --python "$MLSBENCH_PY" \
  "${EXTRA_ARGS[@]}"

echo "[mlsbench-cpu] DONE. summary -> $SUMMARY_JSON"
echo "===== summary.json ====="
cat "$SUMMARY_JSON"

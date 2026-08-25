#!/usr/bin/env bash
# Local (no-slurm, no-apptainer) replacement for
# slurm/cc_eval_mlsbench_cpu_ailab.sh -- the 22-task MLS-Bench CPU eval.
#
#   GPUS=0 MODEL_PATH=<hf_dir_or_id> TAG=<tag> bash scripts/gpublaze/eval_mlsbench_local.sh
#
# What changes vs the Princeton script (semantics preserved otherwise):
#   - MLSBENCH_ROOT defaults to the PATCHED eval harness at
#     .cache/mlsbench-eval (fresh MLS-Bench-dev clone @805adf733 + the
#     FrontierSmith patch layers incl. the view+str_replace+rewrite edit
#     contract; user ruling 2026-08-23 -- the plain dev checkout is NOT valid).
#   - the generated config uses `container_runtime: local` (per-package conda
#     envs) instead of apptainer. `docker` also works on this box if the task
#     images are built. NOTE: root choice = scoring regime; never put numbers
#     from different roots/runtimes in one table.
#   - vLLM serve runs on GPUS via the sesl venv; no GPU keep-alive loop (no
#     auto-cancel policy here).
# Env knobs are the same as the historical script: MLSBENCH_MAX_STEPS,
# MLSBENCH_MAX_TESTS, MLSBENCH_BUDGET_TOKENS, MLSBENCH_USE_REPLACE, CONCURRENCY,
# TASK_TIMEOUT, SMOKE_TASK, TASKS, LIMIT, VLLM_PORT, TP, MAX_MODEL_LEN, ...
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_gpublaze.sh"
PROJECT_ROOT="$FS_ROOT"
cd "$PROJECT_ROOT"

# EXTERNAL_VLLM_URL=http://127.0.0.1:8006/v1 -> attach to an ALREADY-RUNNING
# OpenAI endpoint instead of starting vLLM here (no GPUs claimed). TAG must
# then equal that server's --served-model-name (MLS-dev strips the vllm/
# routing prefix and sends the bare TAG in the request body).
EXTERNAL_VLLM_URL="${EXTERNAL_VLLM_URL:-}"
if [ -n "$EXTERNAL_VLLM_URL" ]; then
  MODEL_PATH="${MODEL_PATH:-external}"
  TAG="${TAG:?with EXTERNAL_VLLM_URL, set TAG to the served-model-name of that server}"
else
  MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:?set MODEL_PATH (HF dir or id)}}"
  TAG="${TAG:-$(basename "$MODEL_PATH")}"
  GPUS="${GPUS:?set GPUS, e.g. GPUS=0}"
  fs_guard_gpus "$GPUS" || exit 1
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# User ruling 2026-08-23: the dev checkout /srv/home/bohanlyu/MLS-Bench is NOT
# a valid eval harness -- it lacks the view+str_replace+rewrite edit contract
# (scripts/mlsbench_edit_contract.diff). The canonical harness is the fresh
# clone + FrontierSmith patch layers at .cache/mlsbench-eval (vendor/data,
# workspace, external_packages symlinked from the dev checkout; conda envs are
# global). Guard below refuses a root without the view contract.
MLSBENCH_ROOT="${MLSBENCH_ROOT:-$FS_ROOT/.cache/mlsbench-eval}"
[ -d "$MLSBENCH_ROOT/src/mlsbench" ] || { echo "ERROR: MLSBENCH_ROOT=$MLSBENCH_ROOT is not an MLS-Bench checkout" >&2; exit 1; }
grep -q 'VIEW_SCHEMA' "$MLSBENCH_ROOT/src/mlsbench/agent/tools.py" 2>/dev/null || {
  echo "ERROR: MLSBENCH_ROOT=$MLSBENCH_ROOT has no view edit contract (mlsbench_edit_contract.diff not applied) -- scores from it are not the Princeton protocol" >&2; exit 1; }
if [ "${USE_REPLACE:-1}" = "1" ] && ! grep -rq -- "use-replace" "$MLSBENCH_ROOT/src/mlsbench" 2>/dev/null; then
  echo "ERROR: MLSBENCH_ROOT has no --use-replace support; every task would fail in <1s" >&2; exit 1
fi
# Record the scoring regime provenance next to the results.
MLS_COMMIT="$(git -C "$MLSBENCH_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
MLS_BRANCH="$(git -C "$MLSBENCH_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_mlsbench_cpu_${TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_BASE/summary.json}"
mkdir -p "$OUTPUT_BASE"
echo "{\"mlsbench_root\": \"$MLSBENCH_ROOT\", \"branch\": \"$MLS_BRANCH\", \"commit\": \"$MLS_COMMIT\", \"container_runtime\": \"${MLSBENCH_CONTAINER_RUNTIME:-local}\"}" > "$OUTPUT_BASE/mlsbench_provenance.json"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
# Host-side LAPACK SIGSEGV guard (see cc_eval_mlsbench_cpu_ailab.sh rationale).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
export EVAL_RESEARCHER_YEAR="${EVAL_RESEARCHER_YEAR:-2026}"
export MLSBENCH_NO_PREBUILT="${MLSBENCH_NO_PREBUILT:-1}"
export MLSBENCH_SCHEDULER_MANAGED=1
export MLSBENCH_USE_REPLACE="${MLSBENCH_USE_REPLACE:-1}"

export VLLM_PORT="${VLLM_PORT:-$(( 34000 + $$ % 9000 ))}"
export HOST=127.0.0.1
AGENT_MODEL="vllm/${TAG}"
SERVED_MODEL_NAME="${TAG} vllm/${TAG}"
PROVIDER_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
[ -n "$EXTERNAL_VLLM_URL" ] && PROVIDER_BASE_URL="$EXTERNAL_VLLM_URL"

export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export DTYPE="${DTYPE:-bfloat16}"
export CONCURRENCY="${CONCURRENCY:-20}"
export TASK_TIMEOUT="${TASK_TIMEOUT:-5400}"

echo "[mlsbench-local] TAG=$TAG MODEL=$MODEL_PATH root=$MLSBENCH_ROOT@$MLS_COMMIT port=$VLLM_PORT"

# ----- generated config: local conda runtime, NO slurm block -------------------
DATA_ROOT="${MLSBENCH_DATA_ROOT:-$MLSBENCH_ROOT/vendor/data}"
SAVE_PATH="${MLSBENCH_SAVE_PATH:-$OUTPUT_BASE/saves}"
mkdir -p "$SAVE_PATH"
GEN_CONFIG="$OUTPUT_BASE/config_vllm_local_gpublaze_$$.yaml"
cat > "$GEN_CONFIG" <<YAML
# AUTO-GENERATED by scripts/gpublaze/eval_mlsbench_local.sh — local runtime, NO slurm block.
max_steps: ${MLSBENCH_MAX_STEPS:-20}
max_tests: ${MLSBENCH_MAX_TESTS:-3}
save_path: ${SAVE_PATH}
data_root: ${DATA_ROOT}
seeds: [42]
container_runtime: ${MLSBENCH_CONTAINER_RUNTIME:-local}
local_thread_limit: ${MLSBENCH_LOCAL_THREADS:-16}

thinking:
  enabled: true
  reasoning_effort: "high"
  budget_tokens: ${MLSBENCH_BUDGET_TOKENS:-10000}

providers:
  vllm:
    api_key: "EMPTY"
    base_url: "${PROVIDER_BASE_URL}"
YAML
echo "[mlsbench-local] generated config: $GEN_CONFIG"; sed 's/^/    /' "$GEN_CONFIG"

if [ -n "$EXTERNAL_VLLM_URL" ]; then
  # Attach-only: never start or stop a server someone else owns.
  curl -fsS "${EXTERNAL_VLLM_URL%/}/models" >/dev/null 2>&1 \
    || { echo "ERROR: external backend ${EXTERNAL_VLLM_URL} not answering /models" >&2; exit 1; }
  echo "[mlsbench-local] using EXTERNAL backend ${EXTERNAL_VLLM_URL} (served='$TAG')"
else
  # ----- start vLLM (tool-calling enabled: hermes parser, required by MLS agent) -
  PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" MODEL_PATH="$MODEL_PATH" \
    setsid bash scripts/start_vllm_server.sh --enable-auto-tool-choice --tool-call-parser hermes &
  VLLM_PID="$!"
  cleanup() { kill -- -"$VLLM_PID" >/dev/null 2>&1 || kill "$VLLM_PID" >/dev/null 2>&1 || true; }
  trap cleanup EXIT INT TERM

  for _ in $(seq 1 900); do
    curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
    sleep 2
    kill -0 "$VLLM_PID" >/dev/null 2>&1 || { echo "vLLM exited early" >&2; exit 1; }
  done
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
  echo "[mlsbench-local] vLLM ready on ${VLLM_PORT}"
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | sed 's/^/    /' || true
fi

# ----- worker pool -------------------------------------------------------------
# The conda base python has mlsbench's deps (openai, yaml); /usr/bin/python3 lacks openai.
MLSBENCH_PY="${MLSBENCH_PY:-/srv/home/bohanlyu/miniconda3/bin/python}"
[ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"

EXTRA_ARGS=()
[ -n "${SMOKE_TASK:-}" ] && EXTRA_ARGS+=(--tasks "$SMOKE_TASK")
# shellcheck disable=SC2086
[ -n "${TASKS:-}" ] && EXTRA_ARGS+=(--tasks $TASKS)
[ -n "${LIMIT:-}" ] && EXTRA_ARGS+=(--limit "$LIMIT")

echo "[mlsbench-local] launching worker pool with $MLSBENCH_PY"
MODEL="$AGENT_MODEL" MLSBENCH_ROOT="$MLSBENCH_ROOT" \
"$MLSBENCH_PY" "$PROJECT_ROOT/scripts/mlsbench_run_cpu_tasks.py" \
  --config "$GEN_CONFIG" \
  --model "$AGENT_MODEL" \
  --root "$MLSBENCH_ROOT" \
  --out "$SUMMARY_JSON" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TASK_TIMEOUT" \
  --python "$MLSBENCH_PY" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
rc=$?
echo "[mlsbench-local] DONE rc=$rc. summary -> $SUMMARY_JSON"
[ -f "$SUMMARY_JSON" ] && cat "$SUMMARY_JSON"
exit $rc

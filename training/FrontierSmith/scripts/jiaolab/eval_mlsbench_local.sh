#!/usr/bin/env bash
# jiaolab (no-slurm, no-apptainer) 22-task MLS-Bench CPU eval.
# Port of scripts/gpublaze/eval_mlsbench_local.sh -- SAME protocol, same env
# knobs, same generated-config shape, same worker pool
# (scripts/mlsbench_run_cpu_tasks.py, unchanged).
#
#   GPUS=2 MODEL_PATH=<hf_dir_or_id> TAG=<tag> bash scripts/jiaolab/eval_mlsbench_local.sh
#   EXTERNAL_VLLM_URL=http://127.0.0.1:8006/v1 TAG=<served-name> bash ...   # attach-only
#
# Machine deltas vs gpublaze, all of them forced by this box:
#   - TOOL-CALL PARSER = qwen3_xml, NOT hermes.  The MLS agent is tool-calling;
#     Qwen3.5 emits the XML tool-call format, and `hermes` ACCEPTS every request
#     while parsing nothing -- a silent 0/22 in ~3 minutes with a green log.
#     The gate below (mls_tools_ok) therefore refuses to start the eval until the
#     backend returns a NON-EMPTY parsed `tool_calls` array; a 200 is not proof.
#   - The harness lives at .cache/mlsbench-eval and its vendor/{data,
#     external_packages,workspace} are REAL directories, not the symlinks into
#     /srv/home/bohanlyu/MLS-Bench that gpublaze uses (no dev checkout here).
#   - container_runtime=local => per-package conda envs `mlsbench-<pkg>` under
#     /home/bohan/miniconda3/envs (shipped from gpublaze and prefix-relocated).
#     scripts/jiaolab/mlsbench_preflight.sh verifies all of that before a GPU is
#     touched.
#   - GPU police: env_jiaolab.sh's dynamic fs_guard_gpus (cards shared with user
#     `druv`; never evict anyone) instead of gpublaze's static 6/7 ban, and
#     GPU_MEMORY_UTILIZATION 0.85 instead of 0.90 so a co-tenant that grows
#     mid-run is not OOM-killed by us. KV-cache size affects throughput only.
#   - TP=1 per card (A100-80G PCIe, no NVLink; Qwen3.5's hybrid attention/GDN
#     stack tensor-parallelises badly over PCIe).
#   - MLSBENCH_PY = the `mlsbench-driver` conda env (python 3.13, numpy/scipy/
#     openai/pyyaml pinned to gpublaze conda-base versions). jiaolab's conda base
#     is a bare python 3.14 with none of those; /usr/bin/python3 has none either.
#     It deliberately does NOT contain causal-learn: gpublaze's driver base has no
#     causallearn either, so causal-observational-linear-gaussian's parser fails to
#     import there and that task records agent_failed. Adding it here would make
#     jiaolab score a task gpublaze does not -- a different regime.
# UNCHANGED from gpublaze (this is the protocol, do not "localise" it):
#   MAX_MODEL_LEN=40960, MAX_NUM_SEQS=32, MAX_NUM_BATCHED_TOKENS=8192,
#   CONCURRENCY=20, TASK_TIMEOUT=5400, max_steps=20, max_tests=3,
#   budget_tokens=10000, reasoning_effort=high, seeds=[42], local_thread_limit=16,
#   EVAL_RESEARCHER_YEAR=2026, MLSBENCH_USE_REPLACE=1, and the 22-task list.
#
# BAN (EVAL_ROBUSTNESS_zh.md 铁律1): jiaolab MLS numbers are NOT same-table
# comparable with gpublaze MLS numbers -- different CPU (Xeon 8358 vs EPYC 9654)
# and the CPU tasks are wall-clock/compute sensitive. Compare jiaolab to jiaolab.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"
PROJECT_ROOT="$FS_ROOT"
cd "$PROJECT_ROOT"

# EXTERNAL_VLLM_URL -> attach to an ALREADY-RUNNING OpenAI endpoint instead of
# starting vLLM here (no GPUs claimed, nothing of somebody else's is stopped).
# TAG must then equal that server's --served-model-name (MLS-dev strips the
# vllm/ routing prefix and sends the bare TAG in the request body).
EXTERNAL_VLLM_URL="${EXTERNAL_VLLM_URL:-}"
if [ -n "$EXTERNAL_VLLM_URL" ]; then
  MODEL_PATH="${MODEL_PATH:-external}"
  TAG="${TAG:?with EXTERNAL_VLLM_URL, set TAG to the served-model-name of that server}"
else
  MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:?set MODEL_PATH (HF dir or id)}}"
  TAG="${TAG:-$(basename "$MODEL_PATH")}"
  GPUS="${GPUS:?set GPUS, e.g. GPUS=2 (cards are shared with user druv -- the guard refuses busy ones)}"
  fs_guard_gpus "$GPUS" || exit 1
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# The canonical harness is the fresh clone + FrontierSmith patch layers, NOT a
# plain MLS-Bench checkout: without the view+str_replace+rewrite edit contract
# (scripts/mlsbench_edit_contract.diff) the scores are a different regime.
MLSBENCH_ROOT="${MLSBENCH_ROOT:-$FS_ROOT/.cache/mlsbench-eval}"
export MLSBENCH_ROOT
export MLSBENCH_PY="${MLSBENCH_PY:-/home/bohan/miniconda3/envs/mlsbench-driver/bin/python}"
[ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"

# container_runtime=local resolves the per-package env through MLS-Bench's
# find_conda_exe(). If that ever returns None the harness SILENTLY falls back to a
# PIP_TARGET site-packages dir -- a different runtime, i.e. different scores with
# no error anywhere. Pin it and put condabin on PATH so there is nothing to guess.
export CONDA_EXE="${CONDA_EXE:-/home/bohan/miniconda3/condabin/conda}"
[ -x "$CONDA_EXE" ] || CONDA_EXE="$(command -v conda || true)"
[ -n "$CONDA_EXE" ] || { echo "ERROR: no conda executable; container_runtime=local needs the mlsbench-<pkg> envs" >&2; exit 1; }
export PATH="$(dirname "$CONDA_EXE"):$PATH"

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  bash "$SCRIPT_DIR/mlsbench_preflight.sh" || {
    echo "ERROR: preflight failed -- refusing to start (SKIP_PREFLIGHT=1 to override)" >&2; exit 1; }
fi
# Kept as hard guards even when preflight is skipped: these two are the contract.
[ -d "$MLSBENCH_ROOT/src/mlsbench" ] || { echo "ERROR: MLSBENCH_ROOT=$MLSBENCH_ROOT is not an MLS-Bench checkout" >&2; exit 1; }
grep -q 'VIEW_SCHEMA' "$MLSBENCH_ROOT/src/mlsbench/agent/tools.py" 2>/dev/null || {
  echo "ERROR: MLSBENCH_ROOT=$MLSBENCH_ROOT has no view edit contract (mlsbench_edit_contract.diff not applied) -- scores from it are not the Princeton protocol" >&2; exit 1; }
if [ "${USE_REPLACE:-1}" = "1" ] && ! grep -rq -- "use-replace" "$MLSBENCH_ROOT/src/mlsbench" 2>/dev/null; then
  echo "ERROR: MLSBENCH_ROOT has no --use-replace support; every task would fail in <1s" >&2; exit 1
fi
MLS_COMMIT="$(git -C "$MLSBENCH_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
MLS_BRANCH="$(git -C "$MLSBENCH_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_mlsbench_cpu_${TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_BASE/summary.json}"
mkdir -p "$OUTPUT_BASE"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
# Host-side LAPACK SIGSEGV guard (see slurm/cc_eval_mlsbench_cpu_ailab.sh).
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
# 0.85, not gpublaze's 0.90: cards are shared with druv (see header).
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export DTYPE="${DTYPE:-bfloat16}"
export CONCURRENCY="${CONCURRENCY:-20}"
export TASK_TIMEOUT="${TASK_TIMEOUT:-5400}"
# Qwen3.5's mamba/GDN layers + vLLM 0.21 prefix caching wedge the engine.
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

echo "{\"mlsbench_root\": \"$MLSBENCH_ROOT\", \"branch\": \"$MLS_BRANCH\", \"commit\": \"$MLS_COMMIT\", \"container_runtime\": \"${MLSBENCH_CONTAINER_RUNTIME:-local}\", \"node\": \"$FS_NODE_NAME\", \"tool_call_parser\": \"$TOOL_CALL_PARSER\"}" > "$OUTPUT_BASE/mlsbench_provenance.json"

echo "[mlsbench-jiaolab] TAG=$TAG MODEL=$MODEL_PATH root=$MLSBENCH_ROOT@$MLS_COMMIT port=$VLLM_PORT parser=$TOOL_CALL_PARSER"

# ----- tool-call PARSE gate ----------------------------------------------------
# Deterministic probe (temperature 0): instruct a test() call and require a
# non-empty parsed tool_calls array. A 200 with content but no tool_calls is
# exactly the hermes-on-Qwen3.5 failure mode -- treat it as NOT ready.
mls_tools_probe() {  # <base_url> <served_name> -> prints the raw response
  curl -sS --max-time "${TOOLS_PROBE_TIMEOUT:-240}" "${1%/}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"temperature\":0,\"max_tokens\":${TOOLS_PROBE_MAX_TOKENS:-4000},\"messages\":[{\"role\":\"system\",\"content\":\"Always respond with a tool call.\"},{\"role\":\"user\",\"content\":\"Run a first experiment now by calling the test tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"test\",\"description\":\"Run a new experiment\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}" \
    2>/dev/null
}
mls_tools_ok() {  # <base_url> <served_name>
  # The body goes through the environment, not through quoting: this snippet has
  # to contain both quote characters and a heredoc keeps it readable.
  MLS_PROBE_BODY="$(mls_tools_probe "$1" "$2")" "$MLSBENCH_PY" - <<'PYGATE'
import json, os, sys

raw = os.environ.get("MLS_PROBE_BODY", "")
try:
    d = json.loads(raw)
    msg = d["choices"][0]["message"]
except Exception as exc:
    # Usually an error body: vLLM rejects a request carrying `tools` outright
    # when it was started without --enable-auto-tool-choice. Show it; the bare
    # KeyError tells nobody anything.
    sys.stderr.write(
        "[tools-gate] no usable response (%s); body: %s\n" % (exc, raw[:400]))
    sys.exit(1)

tc = msg.get("tool_calls") or []
if not tc:
    # The hermes-on-Qwen3.5 failure mode: 200 OK, finish_reason=stop, and the
    # tool call left UNPARSED in content. Print the tail of content so it is
    # obvious the model DID emit one and the server did not parse it.
    content = msg.get("content") or ""
    sys.stderr.write(
        "[tools-gate] response had NO parsed tool_calls -- server started without "
        "--enable-auto-tool-choice, or the wrong --tool-call-parser. "
        "finish_reason=%r content[-300:]=%r\n"
        % (d["choices"][0].get("finish_reason"), content[-300:]))
    sys.exit(1)

sys.stdout.write("[tools-gate] parsed tool_calls: %s\n" % json.dumps(tc))
PYGATE
}

# ----- generated config: local conda runtime, NO slurm block -------------------
DATA_ROOT="${MLSBENCH_DATA_ROOT:-$MLSBENCH_ROOT/vendor/data}"
SAVE_PATH="${MLSBENCH_SAVE_PATH:-$OUTPUT_BASE/saves}"
mkdir -p "$SAVE_PATH"
GEN_CONFIG="$OUTPUT_BASE/config_vllm_local_jiaolab_$$.yaml"
cat > "$GEN_CONFIG" <<YAML
# AUTO-GENERATED by scripts/jiaolab/eval_mlsbench_local.sh — local runtime, NO slurm block.
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
echo "[mlsbench-jiaolab] generated config: $GEN_CONFIG"; sed 's/^/    /' "$GEN_CONFIG"

if [ -n "$EXTERNAL_VLLM_URL" ]; then
  # Attach-only: never start or stop a server someone else owns.
  curl -fsS "${EXTERNAL_VLLM_URL%/}/models" >/dev/null 2>&1 \
    || { echo "ERROR: external backend ${EXTERNAL_VLLM_URL} not answering /models" >&2; exit 1; }
  echo "[mlsbench-jiaolab] using EXTERNAL backend ${EXTERNAL_VLLM_URL} (served='$TAG')"
  # Wait, don't fail immediately: a shared engine may still be warming up.
  WAIT_TOOLS_SEC="${WAIT_TOOLS_SEC:-1800}"
  start=$SECONDS
  until mls_tools_ok "$EXTERNAL_VLLM_URL" "$TAG"; do
    if (( SECONDS - start >= WAIT_TOOLS_SEC )); then
      echo "ERROR: $EXTERNAL_VLLM_URL never returned parsed tool_calls after ${WAIT_TOOLS_SEC}s." >&2
      echo "  Restart that server with: --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER" >&2
      exit 1
    fi
    echo "[mlsbench-jiaolab] backend returns no parsed tool_calls yet; retry in 60s"
    sleep 60
  done
else
  # ----- start vLLM (tool calling ON, qwen3_xml parser: see header) ------------
  PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" MODEL_PATH="$MODEL_PATH" \
    setsid bash scripts/start_vllm_server.sh \
      --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" &
  VLLM_PID="$!"
  cleanup() { kill -- -"$VLLM_PID" >/dev/null 2>&1 || kill "$VLLM_PID" >/dev/null 2>&1 || true; }
  trap cleanup EXIT INT TERM

  for _ in $(seq 1 900); do
    curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
    sleep 2
    kill -0 "$VLLM_PID" >/dev/null 2>&1 || { echo "vLLM exited early" >&2; exit 1; }
  done
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
  echo "[mlsbench-jiaolab] vLLM ready on ${VLLM_PORT}"
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | sed 's/^/    /' || true
  # HARD GATE: our own server, so a missing parser is a bug we must not paper over.
  mls_tools_ok "$PROVIDER_BASE_URL" "$TAG" | tee "$OUTPUT_BASE/tool_call_probe.txt" || {
    echo "ERROR: our vLLM did not return parsed tool_calls with --tool-call-parser $TOOL_CALL_PARSER." >&2
    echo "  Running MLS against it would score a silent 0/22. Aborting." >&2
    exit 1; }
fi

# ----- worker pool -------------------------------------------------------------
EXTRA_ARGS=()
[ -n "${SMOKE_TASK:-}" ] && EXTRA_ARGS+=(--tasks "$SMOKE_TASK")
# shellcheck disable=SC2086
[ -n "${TASKS:-}" ] && EXTRA_ARGS+=(--tasks $TASKS)
[ -n "${LIMIT:-}" ] && EXTRA_ARGS+=(--limit "$LIMIT")

echo "[mlsbench-jiaolab] launching worker pool with $MLSBENCH_PY"
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
echo "[mlsbench-jiaolab] DONE rc=$rc. summary -> $SUMMARY_JSON"
[ -f "$SUMMARY_JSON" ] && cat "$SUMMARY_JSON"
exit $rc

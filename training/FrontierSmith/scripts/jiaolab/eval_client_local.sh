#!/usr/bin/env bash
# jiaolab (no-slurm) CPU client -- one process = one (SOURCE, shard) of the
# FCS/ALE/Research protocol eval, scoring included. Port of
# scripts/gpublaze/eval_client_local.sh.
#
#   TAG=<tag> SOURCE=both NUM_SHARDS=2 SHARD_IDX=0 bash scripts/jiaolab/eval_client_local.sh
#
# Required env: TAG (output naming). MODEL_TAG defaults to TAG; the client polls
# the local registry (scripts/vllm_pool_pick.py) unless VLLM_BASE_URL is set.
# NOTE: with a 2-engine pool, ALWAYS pin VLLM_BASE_URL per shard -- the
# --least-loaded pick races when clients start together and both land on the same
# engine (launch_pool_eval.sh does the pinning for you).
#
# Faithful to the Princeton client: same sampling protocol (32k thinking,
# presence 1.5, n=5, y26 conditioning), same output layout, same judge topology.
# Machine deltas vs gpublaze, all forced by jiaolab:
#   - ALE-Bench judging runs on the APPTAINER backend
#     (ALE_BENCH_CONTAINER_BACKEND=apptainer, SIFs in $ALE_BENCH_APPTAINER_DIR).
#     docker here needs sudo (unavailable) and there is no bwrap, so neither the
#     harness's native docker backend nor gpublaze's `host` bwrap backend can
#     run. scripts/jiaolab/pysite/ale_apptainer_backend.py supplies an EAGER,
#     CPU-PINNED apptainer sandbox so infra failures raise AleInfraError instead
#     of being scored as silent-zero COMPILATION_ERRORs.
#   - CONCURRENCY defaults to 64 and REQUEST_TIMEOUT to 7200: verified settings
#     for one TP=1 A100 engine per client shard at 32k max_tokens.
#   - judge_node_meta.json node class here is jiaolab (A100-80G PCIe host). Per
#     EVAL_ROBUSTNESS_zh.md 铁律1, scores judged on this box are NOT same-table
#     comparable with gpublaze (H100) or the ailab/pli history -- every jiaolab
#     comparison must be against a jiaolab anchor run.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"
PROJECT_ROOT="$FS_ROOT"
cd "$PROJECT_ROOT"

: "${TAG:?set TAG}"
MODEL_TAG="${MODEL_TAG:-$TAG}"
SOURCE="${SOURCE:-both}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
REGISTRY="$VLLM_POOL_REGISTRY"
PYBIN="$FS_CLIENT_VENV/bin/python"

# ---- find a backend ----------------------------------------------------------
POOL_WAIT="${POOL_WAIT:-3600}"
if [ -z "${VLLM_BASE_URL:-}" ]; then
  wait_start=$SECONDS
  while :; do
    VLLM_BASE_URL="$("$PYBIN" scripts/vllm_pool_pick.py --tag "$MODEL_TAG" --least-loaded 2>/dev/null)" && break
    elapsed=$(( SECONDS - wait_start ))
    [ "$elapsed" -ge "$POOL_WAIT" ] && { echo "ERROR: no live backend for tag=$MODEL_TAG after ${elapsed}s" >&2; exit 1; }
    echo "[client-local] waiting for backend tag=$MODEL_TAG (${elapsed}s/${POOL_WAIT}s)"; sleep 15
  done
fi
export VLLM_BASE_URL
export SERVED_MODEL_NAME="$MODEL_TAG"
echo "[client-local] TAG=$TAG src=$SOURCE shard=$SHARD_IDX/$NUM_SHARDS backend=$VLLM_BASE_URL"

# ---- client heartbeat (keeps an IDLE_EXIT=1 serve_local.sh alive) -------------
_hp="${VLLM_BASE_URL#http://}"; _hp="${_hp%%/*}"
HB_DIR="$REGISTRY/${MODEL_TAG}__${_hp%%:*}__${_hp##*:}.json.clients"
HB_FILE=""
mkdir -p "$HB_DIR" 2>/dev/null || true
if [ -d "$HB_DIR" ]; then
  HB_FILE="$HB_DIR/local-$$"
  touch "$HB_FILE"
  ( while :; do touch "$HB_FILE" 2>/dev/null || true; sleep 60; done ) & HB_PID=$!
fi

cleanup() {
  [ -n "${HB_PID:-}" ]    && kill "$HB_PID"    >/dev/null 2>&1 || true
  [ -n "${HB_FILE:-}" ]   && touch "$HB_FILE"  >/dev/null 2>&1 || true
  [ -n "${JUDGE_PID:-}" ] && kill "$JUDGE_PID" >/dev/null 2>&1 || true
  case "${RUNTIME_DIR:-}" in
    "$PROJECT_ROOT"/.cache/frontiercs-judge-*) rm -rf "$RUNTIME_DIR" >/dev/null 2>&1 || true ;;
  esac
  [ -n "${GJ_CGROUP_PREFIX:-}" ] && rm -rf "${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ---- judge (non-research sources) ---------------------------------------------
if [ "$SOURCE" != "research" ]; then
  read -r _P1 _P2 <<< "$("$PYBIN" - <<'PY'
import socket
socks = []
for _ in range(2):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0)); socks.append(s)
print(*[s.getsockname()[1] for s in socks])
for s in socks: s.close()
PY
)"
  export PORT="${JUDGE_PORT:-$_P1}"
  export GJ_PORT="${GJ_PORT:-$_P2}"
  export GJ_CGROUP_PREFIX="gojudge-local-$$"
  export RUNTIME_DIR="$PROJECT_ROOT/.cache/frontiercs-judge-cpucli-local-$$"
  export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"
  bash "$SCRIPT_DIR/start_frontiercs_judge_local.sh" &
  JUDGE_PID=$!
  ok=0
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 && \
    curl -fsS "http://127.0.0.1:${PORT}/health"    >/dev/null 2>&1 && { ok=1; break; }
    sleep 0.5
    kill -0 "$JUDGE_PID" 2>/dev/null || { echo "judge exited early" >&2; exit 1; }
  done
  [ "$ok" = 1 ] || { echo "judge never healthy" >&2; exit 1; }
  echo "[client-local] judge ready (node $PORT / go-judge $GJ_PORT)"
fi

# ---- ALE-Bench judge backend: apptainer (the only sandbox on this box) --------
# See scripts/jiaolab/pysite/ale_apptainer_backend.py for the full
# docker-flag -> apptainer-flag equivalence map and the failure contract.
export ALE_BENCH_CONTAINER_BACKEND="${ALE_BENCH_CONTAINER_BACKEND:-apptainer}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR/pysite"
# The historical driver preflight requires this path to EXIST; here it is also
# the real SIF directory.
mkdir -p "$ALE_BENCH_APPTAINER_DIR"
if [ "$ALE_BENCH_CONTAINER_BACKEND" = "apptainer" ] && [ "$SOURCE" != "research" ]; then
  command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not on PATH; ALE judging cannot run" >&2; exit 1; }
  for _sif in "$ALE_BENCH_APPTAINER_DIR/ale-bench_cpp17-202301.sif" "$ALE_BENCH_APPTAINER_DIR/rust_1.79.0-buster.sif"; do
    [ -f "$_sif" ] || { echo "ERROR: missing SIF $_sif -- refusing to score ALE with an incomplete toolchain" >&2; exit 1; }
  done
fi

# ---- preflight: official Frontier-CS python package -------------------------
# scripts/eval_qwen35_base_vllm_request.py adds $PROJECT_ROOT/.cache/Frontier-CS-official
# (+ /src) to sys.path and imports `algorithmic.scripts.generate_solutions`
# (official prompt + extract_cpp_code) and `frontier_cs.runner.algorithmic_local`.
# With --frontiercs-score-backend official (the default, and the protocol) a
# missing clone does NOT abort: it raises inside _score for EVERY frontiercs
# sample, each one is recorded as an error with reward 0.0, and the run silently
# produces a floor-zero FCS number. Verified failure on 2026-08-25 (15 samples
# burned before it was caught), so check it up front and refuse to start.
if [ "$SOURCE" != "research" ] && [ "$SOURCE" != "alebench" ]; then
  _OFF="$PROJECT_ROOT/.cache/Frontier-CS-official"
  for _p in "$_OFF/algorithmic/scripts/generate_solutions.py" "$_OFF/src/frontier_cs/runner/algorithmic_local.py"; do
    [ -f "$_p" ] || { echo "ERROR: missing official Frontier-CS asset: $_p" >&2
      echo "  .cache/Frontier-CS-official must point at the official clone (rsync it from gpublaze:" >&2
      echo "  .cache/external/Frontier-CS, then ln -sfn it). Refusing to start -- otherwise every" >&2
      echo "  frontiercs sample scores a fake 0.0." >&2; exit 1; }
  done
fi

# ---- eval configuration (protocol identical to cc_eval_cpu_client.sh) ---------
export EVAL_DECOUPLE=1 RESUME="${RESUME:-1}"
export CONCURRENCY="${CONCURRENCY:-64}"          # one TP=1 A100 engine per shard
export EVAL_SCORE_CONCURRENCY="${EVAL_SCORE_CONCURRENCY:-4}"  # 4-way judging; each ALE sandbox leases 1 core
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0.0}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export N_SAMPLES="${N_SAMPLES:-5}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [ "$MAX_TOKENS" -lt 32768 ]; then
  echo "ERROR: MAX_TOKENS=$MAX_TOKENS < 32768 would truncate generation and is not the protocol" >&2
  exit 1
fi
if [ "${EVAL_RESEARCHER_YEAR-__unset__}" = "__unset__" ]; then EVAL_RESEARCHER_YEAR=2026; fi
[ "$EVAL_RESEARCHER_YEAR" = "off" ] && EVAL_RESEARCHER_YEAR=""
export EVAL_RESEARCHER_YEAR
export EVAL_SYS_PROMPT_MODE="${EVAL_SYS_PROMPT_MODE:-full}"
echo "[client-local] protocol: max_tokens=$MAX_TOKENS temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K pp=$PRESENCE_PENALTY n=$N_SAMPLES"

case "$SOURCE" in
  both)     suf="thinking_32k_both_vllm" ;;
  research) suf="research_thinking_32k_vllm" ;;
  alebench) suf="ale40_thinking_32k_vllm" ;;
  *)        suf="${SOURCE}_thinking_32k_vllm" ;;
esac
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_${TAG}_${suf}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/shard_$SHARD_IDX}"
export SAMPLES_JSONL="${SAMPLES_JSONL:-$OUTPUT_DIR/samples.jsonl}"
export SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_DIR/summary_shard.json}"
EXTRA_DRIVER_ARGS=()
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
EXTRA_DRIVER_ARGS+=(--timeout "$REQUEST_TIMEOUT")
if [ "$SOURCE" = "research" ]; then
  export MAX_ERRORS="${MAX_ERRORS:-${RESEARCH_MAX_ERRORS:-30}}"
  export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-$PYBIN}"
  export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-0}"
  export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$PROJECT_ROOT/.cache/julia_depot}"
  export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-$PROJECT_ROOT/.cache/julia_env}"
  RESEARCH_DATA="${RESEARCH_DATA:-$PROJECT_ROOT/data/frontiercs/research_cpu.parquet}"
  EXTRA_DRIVER_ARGS+=(--research-data "$RESEARCH_DATA")
else
  export MAX_ERRORS="${MAX_ERRORS:-0}"
fi
export SOURCE NUM_SHARDS SHARD_IDX

# ---- judge-node metadata (timing audit convention, sans slurm) ----------------
mkdir -p "$OUTPUT_DIR"
NODE_META="$OUTPUT_DIR/judge_node_meta.json"
FS_CALIB_JSON="$(timeout 60 "$PYBIN" scripts/gojudge_shim_v2.py -calibrate-only 2>/dev/null | sed -n '/^{/,$p' || echo null)" \
FS_CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')" \
FS_JUDGE_VARIANT="jiaolab_local_${GJ_BACKEND:-auto}" NODE_META="$NODE_META" \
FS_ALE_BACKEND="${ALE_BENCH_CONTAINER_BACKEND}" \
"$PYBIN" - <<'PYEOF'
import json, os
calib = None
try:
    calib = json.loads(os.environ.get("FS_CALIB_JSON") or "null")
except Exception:
    pass
meta = {
    "node": os.uname().nodename,
    "slurm_job_id": None,
    "partition": "jiaolab-local",
    "cpu_model": os.environ.get("FS_CPU_MODEL"),
    "judge_variant": os.environ.get("FS_JUDGE_VARIANT"),
    "ale_container_backend": os.environ.get("FS_ALE_BACKEND"),
    "shim_pin_cores": os.environ.get("SHIM_PIN_CORES"),
    "source": os.environ.get("SOURCE"),
    "node_speed_calibration": calib,
}
with open(os.environ["NODE_META"], "w") as f:
    json.dump(meta, f, indent=1)
PYEOF
echo "[client-local] node meta -> $NODE_META"

echo "[client-local] output -> $OUTPUT_DIR (max_errors=$MAX_ERRORS resume=$RESUME)"
bash scripts/eval_base_model_qwen35_9b_vllm_request.sh "${EXTRA_DRIVER_ARGS[@]}"
rc=$?
echo "[client-local] DONE rc=$rc"
exit $rc

#!/bin/bash
#SBATCH --job-name=cc-eval-cpu-client
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
#
# CPU half of the split eval architecture (user ruling 2026-08-16: GPU jobs only
# serve; generation requests + ALL scoring run here). One job = one
# (SOURCE, shard). Scoring is CPU-bound by construction: FCS = go-judge
# compiling/running C++, Research = CPU subset, ALE = simulator. The client
# itself is light (HTTP + aggregation), so the default core count is small
# (-c 16) to slot into the cpu partition easily; scoring gets the extra cores
# vs. the old co-located 8.
#
# Discovery: polls the GPFS registry (scripts/vllm_pool_pick.py) for a backend
# serving MODEL_TAG, for up to POOL_WAIT seconds, so client/server start order
# does not matter. Once attached it touches
#   $REGISTRY/<entry>.json.clients/<this job id>
# every 60s -- that heartbeat is what keeps a cc_serve_only.sh backend alive;
# when all clients stop touching for 15 min the server releases its GPUs.
#
# Routing: compute nodes reach each other directly over TCP (verified, jobs
# 12515903/12515916); no login-node proxy.
#
# Required env: TAG (output naming), and MODEL_TAG (served model name; defaults
# to TAG) or an explicit VLLM_BASE_URL.
# Optional: SOURCE (both|research|alebench, default both), NUM_SHARDS, SHARD_IDX,
#           POOL_WAIT, OUTPUT_BASE/OUTPUT_DIR, FRONTIERCS_LIMIT, ALEBENCH_LIMIT.
set -uo pipefail
PROJECT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$PROJECT_ROOT"

: "${TAG:?set TAG}"
MODEL_TAG="${MODEL_TAG:-$TAG}"
SOURCE="${SOURCE:-both}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
REGISTRY="${VLLM_POOL_REGISTRY:-$PROJECT_ROOT/.cache/vllm_pool}"

# ---- find a backend (poll + log; the backend may still be loading weights) ----
POOL_WAIT="${POOL_WAIT:-14400}"   # 4h: covers the serve job's own pli queue time (measured ~40-65 min)
if [ -z "${VLLM_BASE_URL:-}" ]; then
  wait_start=$SECONDS
  while :; do
    VLLM_BASE_URL="$(python3 scripts/vllm_pool_pick.py --tag "$MODEL_TAG" --least-loaded 2>/dev/null)" && break
    elapsed=$(( SECONDS - wait_start ))
    if [ "$elapsed" -ge "$POOL_WAIT" ]; then
      echo "ERROR: no live backend for tag=$MODEL_TAG after ${elapsed}s (POOL_WAIT=$POOL_WAIT)" >&2
      exit 1
    fi
    echo "[cpu-client] waiting for backend tag=$MODEL_TAG (${elapsed}s / ${POOL_WAIT}s)"
    sleep 30
  done
fi
export VLLM_BASE_URL
echo "[cpu-client] TAG=$TAG src=$SOURCE shard=$SHARD_IDX/$NUM_SHARDS backend=$VLLM_BASE_URL cpus=${SLURM_CPUS_PER_TASK:-?}"

# The model name in each request must match the backend's --served-model-name,
# or vLLM answers 404.
export SERVED_MODEL_NAME="$MODEL_TAG"

# ---- client heartbeat (keeps a serve-only backend alive while we use it) ------
# Entry filename is the registry convention TAG__host__port.json; derive it from
# the URL we picked so this also works when VLLM_BASE_URL was passed explicitly.
_hp="${VLLM_BASE_URL#http://}"; _hp="${_hp%%/*}"
BACKEND_HOST="${_hp%%:*}"; BACKEND_PORT="${_hp##*:}"
HB_FILE=""
HB_DIR="$REGISTRY/${MODEL_TAG}__${BACKEND_HOST}__${BACKEND_PORT}.json.clients"
mkdir -p "$HB_DIR" 2>/dev/null || true
if [ -d "$HB_DIR" ]; then
  HB_FILE="$HB_DIR/${SLURM_JOB_ID:-$$}"
  touch "$HB_FILE"
  ( while :; do touch "$HB_FILE" 2>/dev/null || true; sleep 60; done ) &
  HB_PID=$!
  echo "[cpu-client] heartbeat -> $HB_FILE (every 60s)"
else
  echo "[cpu-client] WARNING: cannot create $HB_DIR; a serve-only backend may idle-exit under us" >&2
fi

cleanup() {
  [ -n "${HB_PID:-}" ]    && kill "$HB_PID"    >/dev/null 2>&1 || true
  # Leave the heartbeat FILE behind, with a final touch marking end-of-work. A
  # client can attach and finish entirely between two of the server's 5-min
  # censuses (measured 2026-08-20: a 4.7-min resume-backfill client was never
  # seen by the server), and deleting the file would erase the only evidence it
  # was ever there. Staleness (15 min) is the exit signal either way; the serve
  # job rm -rf's the whole .clients dir when it exits.
  [ -n "${HB_FILE:-}" ]   && touch "$HB_FILE"  >/dev/null 2>&1 || true
  [ -n "${JUDGE_PID:-}" ] && kill "$JUDGE_PID" >/dev/null 2>&1 || true
  # judge workdir is ~9k files; leaking it per run is what filled the shared
  # fileset to 99.8M/100M inodes.
  case "${RUNTIME_DIR:-}" in
    "$PROJECT_ROOT"/.cache/frontiercs-judge-*) rm -rf "$RUNTIME_DIR" >/dev/null 2>&1 || true ;;
  esac
  [ -n "${GJ_CGROUP_PREFIX:-}" ] && rm -rf "${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ---- judge (pure CPU, local to this job; needed for FCS/ALE scoring) ----------
if [ "$SOURCE" != "research" ]; then
  # Same probe-don't-hash rule as the vLLM port: ask the OS for two free ports.
  read -r _P1 _P2 <<< "$(python3 -c '
import socket
socks = []
for _ in range(2):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    socks.append(s)
print(*[s.getsockname()[1] for s in socks])
for s in socks:
    s.close()
')"
  export PORT="${JUDGE_PORT:-$_P1}"
  export GJ_PORT="${GJ_PORT:-$_P2}"
  export GJ_CGROUP_PREFIX="gojudge-${SLURM_JOB_ID:-$$}"
  export RUNTIME_DIR="$PROJECT_ROOT/.cache/frontiercs-judge-cpucli-${SLURM_JOB_ID:-manual}"
  export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"
  export GJ_BACKEND="${GJ_BACKEND:-auto}"

  scripts/start_frontiercs_judge_hybrid.sh &
  JUDGE_PID=$!
  ok=0
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 && \
    curl -fsS "http://127.0.0.1:${PORT}/health"    >/dev/null 2>&1 && { ok=1; break; }
    sleep 0.5
    kill -0 "$JUDGE_PID" 2>/dev/null || { echo "judge exited early" >&2; exit 1; }
  done
  [ "$ok" = 1 ] || { echo "judge never healthy" >&2; exit 1; }
  echo "[cpu-client] judge ready (node $PORT / go-judge $GJ_PORT backend=$GJ_BACKEND)"
fi

# ---- eval configuration --------------------------------------------------------
export EVAL_DECOUPLE=1 RESUME="${RESUME:-1}"
export CONCURRENCY="${CONCURRENCY:-96}"
# Scoring gets more cores here than in the co-located job (16 dedicated vs 8
# shared with vLLM), so the scorer pool default goes up 8 -> 12.
export EVAL_SCORE_CONCURRENCY="${EVAL_SCORE_CONCURRENCY:-12}"

# THE SAMPLING PROTOCOL. These are not tunables -- they define the measurement,
# and every published number in this campaign uses them. The eval driver's own
# defaults are NOT these (notably --max-tokens defaults to 16000), so a launcher
# that forgets to set them silently produces a different, much worse benchmark:
# the first run of this script scored rlv12_base_s20 at FCS 4.357 (vs 6.816 for
# the untrained base) purely because generation was cut at 16000 tokens -- 89% of
# samples scored 0 with completion_tokens pinned at exactly 16000.
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0.0}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export N_SAMPLES="${N_SAMPLES:-5}"
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
# Fail loudly rather than silently measuring something else.
if [ "$MAX_TOKENS" -lt 32768 ]; then
  echo "ERROR: MAX_TOKENS=$MAX_TOKENS < 32768 would truncate generation and is not the protocol" >&2
  exit 1
fi

# Time-conditioning protocol (user design): TRAINING system prompts carry each
# method's real historical year ("It is now year 1952...", verified across all
# 2590 corpus records, span 1690-2026); EVALUATION always conditions on the
# present. This env var was silently unset in every earlier eval launcher, so
# evals ran with NO system message and the eval half of the design was dropped.
# Measured impact of restoring it (A/B on rlv12 soup arms, n=850/arm): none
# beyond noise on FCS -- so restoring it does not invalidate earlier numbers,
# but from here on the protocol is applied as designed. EVAL_SYS_PROMPT_MODE
# defaults to the FULL training template (incl. the delivery clause) so the
# eval-time prompt is byte-identical to the training-time one.
# ${VAR:-default} treats EMPTY as unset, so a caller passing EVAL_RESEARCHER_YEAR=
# to mean "no system prompt" silently got 2026 instead -- the b48k budget runs were
# contaminated exactly this way (base's -1.6 "budget effect" was the prompt cost).
# Use the unset-only form and an explicit "off" switch.
if [ "${EVAL_RESEARCHER_YEAR-__unset__}" = "__unset__" ]; then EVAL_RESEARCHER_YEAR=2026; fi
[ "$EVAL_RESEARCHER_YEAR" = "off" ] && EVAL_RESEARCHER_YEAR=""
export EVAL_RESEARCHER_YEAR
export EVAL_SYS_PROMPT_MODE="${EVAL_SYS_PROMPT_MODE:-full}"
echo "[cpu-client] protocol: max_tokens=$MAX_TOKENS temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K pp=$PRESENCE_PENALTY n=$N_SAMPLES"

# ---- output layout + error budget (same conventions as the selfhosted job) ----
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
# Research evaluators time out / flake as a matter of course (2400s evaluator
# timeouts are even a model-quality signal), so strict --max-errors=0 kills the
# whole phase on the first flake. FCS/ALE keep 0 (their infra errors mean real
# breakage).
if [ "$SOURCE" = "research" ]; then
  export MAX_ERRORS="${MAX_ERRORS:-${RESEARCH_MAX_ERRORS:-30}}"
else
  export MAX_ERRORS="${MAX_ERRORS:-0}"
fi
export SOURCE NUM_SHARDS SHARD_IDX


# ---- judge-node timing-fidelity metadata (timing audit 2026-08-19/20) ---------
# FCS/ALE judging is timing-sensitive: cpu-partition genoa nodes measure 15-20%
# slower than the ailab/pli hosts all historical numbers were judged on --
# enough to flip near-threshold solutions 100->0 (concurrency <=2x was measured
# innocent; the node class is the variable). Two defenses, per the audit:
#   1. pin the node class at submit time (sbatch --constraint=...; the split
#      submitter forwards CLIENT_CONSTRAINT) when comparing against history;
#   2. ALWAYS record the node identity + speed calibration next to the shard''s
#      samples.jsonl/summary so any cross-node comparison can be audited later.
mkdir -p "$OUTPUT_DIR"
NODE_META="$OUTPUT_DIR/judge_node_meta.json"
FS_CALIB_JSON="$(timeout 60 "$PROJECT_ROOT/.venv/bin/python" scripts/gojudge_shim_v2.py -calibrate-only 2>/dev/null | sed -n '/^{/,$p' || echo null)" \
FS_NODE_FEATURES="$(scontrol show node "$(hostname -s)" 2>/dev/null | grep -o 'AvailableFeatures=[^ ]*' | cut -d= -f2)" \
FS_CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')" \
FS_JUDGE_VARIANT="gojudge_shim_v1_unpinned" NODE_META="$NODE_META" \
python3 - <<'PYEOF'
import json, os
calib = None
try:
    calib = json.loads(os.environ.get("FS_CALIB_JSON") or "null")
except Exception:
    pass
meta = {
    "node": os.uname().nodename,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "node_features": os.environ.get("FS_NODE_FEATURES"),
    "cpu_model": os.environ.get("FS_CPU_MODEL"),
    "judge_variant": os.environ.get("FS_JUDGE_VARIANT"),
    "shim_pin_cores": os.environ.get("SHIM_PIN_CORES"),
    "source": os.environ.get("SOURCE"),
    "node_speed_calibration": calib,
}
with open(os.environ["NODE_META"], "w") as f:
    json.dump(meta, f, indent=1)
PYEOF
echo "[cpu-client] node meta -> $NODE_META"

echo "[cpu-client] output -> $OUTPUT_DIR (max_errors=$MAX_ERRORS resume=$RESUME)"
bash scripts/eval_base_model_qwen35_9b_vllm_request.sh
rc=$?
echo "[cpu-client] DONE rc=$rc"
exit $rc

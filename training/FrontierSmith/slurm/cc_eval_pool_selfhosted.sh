#!/bin/bash
#SBATCH --job-name=cc-eval-pool-self
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
#
# One job that both SERVES the model and RUNS eval clients against it, plus
# publishes the backend so other jobs can share it.
#
# Why this shape rather than the pure split: on this cluster the `cpu` partition
# is priority-bound (Reason=Priority, not lack of cores), so standalone CPU
# clients sat for hours while GPU-side backends burned idle. Meanwhile each GPU
# job already carries 8 cores/GPU that vLLM barely uses -- vLLM's own CPU work is
# a couple of cores; the rest sat idle. So we co-locate the scoring work with the
# server it talks to, and STILL register the backend so extra CPU-only clients
# can attach whenever the cpu queue frees up. Best of both: nothing idles waiting
# for a scheduler, and capacity is still shareable.
#
# Required env: MODEL, TAG. Optional: KIND (both|fcsale|research), SHARDS.
set -uo pipefail
PROJECT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$PROJECT_ROOT"
: "${MODEL:?set MODEL}" "${TAG:?set TAG}"
KIND="${KIND:-both}"
SHARDS="${SHARDS:-1}"

# ---- serve (bind 0.0.0.0 so this backend is shareable, and register it) -------
REGISTRY="${VLLM_POOL_REGISTRY:-$PROJECT_ROOT/.cache/vllm_pool}"
mkdir -p "$REGISTRY"
# Port selection must PROBE, not hash. Deriving it as job_id % N looks unique but
# collides whenever two concurrent jobs' ids are N apart -- job 12517854 died with
# "Address already in use" for exactly that reason, and the odds get worse the
# more arms run at once. Ask the OS for a free port instead, then keep it: bind
# with SO_REUSEADDR, read back the assigned port, close, and hand it to vLLM.
export VLLM_PORT="${VLLM_PORT:-$(python3 -c '
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 0))
print(s.getsockname()[1])
s.close()
')}"
HOSTN="$(hostname -s)"
ENTRY="$REGISTRY/${TAG}__${HOSTN}__${VLLM_PORT}.json"

# All the throughput knobs, same as the standalone pool server.
export TP="${TP:-2}" ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FS_VLLM_PENALTY_FASTPATH=1
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"

# Same probe-don't-hash rule for the judge's two ports.
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
export RUNTIME_DIR="$PROJECT_ROOT/.cache/frontiercs-judge-pool-${SLURM_JOB_ID:-manual}"
export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"
export GJ_BACKEND="${GJ_BACKEND:-auto}"

cleanup() {
  rm -f "$ENTRY" >/dev/null 2>&1 || true
  [ -n "${JUDGE_PID:-}" ] && kill "$JUDGE_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}"  ] && kill "$VLLM_PID"  >/dev/null 2>&1 || true
  # judge workdir is ~9k files; leaking it per run is what filled the shared
  # fileset to 99.8M/100M inodes.
  case "$RUNTIME_DIR" in
    "$PROJECT_ROOT"/.cache/frontiercs-judge-*) rm -rf "$RUNTIME_DIR" >/dev/null 2>&1 || true ;;
  esac
  rm -rf "${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[self] TAG=$TAG on ${HOSTN}:${VLLM_PORT} TP=$TP prefix=$ENABLE_PREFIX_CACHING seqs=$MAX_NUM_SEQS"
HOST=0.0.0.0 PORT="$VLLM_PORT" MODEL_PATH="$MODEL" SERVED_MODEL_NAME="$TAG" \
  bash scripts/start_vllm_server.sh &
VLLM_PID=$!

ready=0
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "vLLM never ready" >&2; exit 1; }

# Publish only after it answers, atomically (temp file in the same dir + rename)
# so a reader can never see a partial entry or a not-yet-loaded backend.
tmp="$(mktemp "$REGISTRY/.tmp.XXXXXX")"
printf '{"tag":"%s","host":"%s","port":%s,"url":"http://%s:%s/v1","model":"%s","job_id":"%s"}\n' \
  "$TAG" "$HOSTN" "$VLLM_PORT" "$HOSTN" "$VLLM_PORT" "$MODEL" "${SLURM_JOB_ID:-manual}" > "$tmp"
mv -f "$tmp" "$ENTRY"
echo "[self] registered $ENTRY"
( while kill -0 "$VLLM_PID" 2>/dev/null; do touch "$ENTRY" 2>/dev/null || true; sleep 60; done ) &

# ---- judge (needed for FCS/ALE scoring) --------------------------------------
if [ "$KIND" != "research" ]; then
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
  echo "[self] judge ready"
fi

# ---- run the eval against our own backend ------------------------------------
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export SERVED_MODEL_NAME="$TAG"
export EVAL_DECOUPLE=1 RESUME="${RESUME:-1}"
export CONCURRENCY="${CONCURRENCY:-96}"
export EVAL_SCORE_CONCURRENCY="${EVAL_SCORE_CONCURRENCY:-8}"

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
echo "[self] protocol: max_tokens=$MAX_TOKENS temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K pp=$PRESENCE_PENALTY n=$N_SAMPLES"

run_kind () {  # source_name output_suffix
  local src="$1" suf="$2"
  local ob="$PROJECT_ROOT/outputs/cc_eval_${TAG}_${suf}"
  # Research evaluators time out / flake as a matter of course (2400s evaluator
  # timeouts are even a model-quality signal), so strict --max-errors=0 kills the
  # whole phase on the first flake -- job 12638063 died 2 samples in on a spurious
  # cwd error. FCS/ALE keep 0 (their infra errors mean real breakage).
  local maxerr=0
  [ "$src" = "research" ] && maxerr="${RESEARCH_MAX_ERRORS:-30}"
  for shard in $(seq 0 $((SHARDS-1))); do
    echo "[self] eval src=$src shard=$shard/$SHARDS maxerr=$maxerr"
    SOURCE="$src" NUM_SHARDS="$SHARDS" SHARD_IDX="$shard" \
    MAX_ERRORS="$maxerr" \
    OUTPUT_DIR="$ob/shard_$shard" \
    SAMPLES_JSONL="$ob/shard_$shard/samples.jsonl" \
    SUMMARY_JSON="$ob/shard_$shard/summary_shard.json" \
      bash scripts/eval_base_model_qwen35_9b_vllm_request.sh
  done
}

case "$KIND" in
  fcsale)   run_kind both     thinking_32k_both_vllm ;;
  research) run_kind research research_thinking_32k_vllm ;;
  # ALE alone, for the 40-problem set. The shipped val.parquet holds only 10 of
  # ALE-Bench's 40 problems, which is the sole reason ALE's difference-SEM is
  # ~96 points and every cross-arm ALE claim so far has been unusable. Point
  # ALEBENCH_VAL_DATA at data/alebench/full40.parquet and use this KIND; the
  # output suffix keeps the two problem sets in separate directories so a 40-
  # problem number can never be silently compared against a 10-problem one.
  # KIND=ale runs ONLY ale (fcsale/both already include the same 40-problem set).
  ale)      run_kind alebench ale40_thinking_32k_vllm ;;
  both)     run_kind both     thinking_32k_both_vllm
            run_kind research research_thinking_32k_vllm ;;
esac
echo "[self] DONE"

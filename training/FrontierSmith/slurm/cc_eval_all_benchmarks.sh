#!/usr/bin/env bash
# =============================================================================
# ONE self-contained Slurm job that evaluates a single model on ALL benchmarks
# (FrontierCS + ALE-Bench + MLS-Bench + Theta + TTT) by serving the model via
# vLLM **ONCE** and running every benchmark against that single shared server,
# then aggregating all scores into one summary_all.json.
#
# This lets us evaluate one (model, soup-alpha) on EVERYTHING in a single job,
# instead of standing up a separate vLLM per benchmark.
#
# DESIGN
#   1. Serve the model via vLLM ONCE on $VLLM_PORT (reuses
#      scripts/start_vllm_server.sh). The server is launched with the MLS-Bench
#      tool-call parser flags (--enable-auto-tool-choice --tool-call-parser
#      hermes); those are harmless for FCS/ALE/Theta (which never send tools).
#      We register MULTIPLE served-model names so every benchmark's client hits
#      one:  cc-<TAG>-think  (FCS/ALE/Theta config)  +  <TAG>  +  vllm/<TAG> (MLS).
#      Health-gate on /health + /v1/models before any benchmark runs.
#   2. Start the FrontierCS go-judge + ALE apptainer ONCE (reuses
#      scripts/start_frontiercs_judge_hybrid.sh, the exact same judge-start logic
#      cc_eval_thinking_both_ailab.sh uses) into the now-stable memory state.
#   3. Run each benchmark against that shared server, EACH FAIL-SOFT (own
#      subshell + logfile; on failure log + record score="FAILED" and continue).
#      One benchmark dying must NOT abort the others.
#        FCS  + ALE   -> scripts/eval_base_model_qwen35_9b_vllm_request.sh
#        MLS          -> scripts/mlsbench_run_cpu_tasks.py (local apptainer pool)
#        Theta (5)    -> OpenEvolve loop, per task, faithful config, objective_value
#        TTT (AC1/AC2/circle) -> same OpenEvolve engine, TTT-faithful task subset
#   4. Aggregate EVERY score into ONE outputs/cc_eval_all_<TAG>/summary_all.json.
#
# PER-BENCHMARK 口径 == the current (restored) one:
#   FCS  : strip-think extract, official prompt+scorer, thinking-general decode.
#   ALE  : performance.mean@5.
#   Theta: CONFIG_VARIANT=faithful, read best_objective_value (NOT combined_score).
#   TTT  : AC1 (first_autocorr) + AC2 (second_autocorr) + circle only.
#
# ENV INTERFACE
#   MODEL_PATH   (required-ish; default models/Qwen3.5-9B)
#   TAG          (default basename MODEL_PATH)
#   BENCHMARKS   space/comma list to select a subset; default "fcs ale mls theta ttt".
#                (fcs+ale always run together in one process; listing either runs both.)
#   VALIDATE=1   small/fast smoke: few FCS/ALE problems, 1 Theta task, few iters.
#   SKIP_MLS=1   skip the slow MLS-Bench block (also implied when 'mls' not in BENCHMARKS).
#   Passthrough (same meaning as the per-benchmark launchers):
#     CONCURRENCY, FRONTIERCS_STRIP_THINK_EXTRACT, MAX_TOKENS, N_SAMPLES,
#     FRONTIERCS_LIMIT, ALEBENCH_LIMIT, THETA_ITERATIONS, THETA_TASKS, TTT_TASKS,
#     MLSBENCH_ROOT, EVAL_RESEARCHER_YEAR.
#
# RESOURCE ASK (see #SBATCH below): 1 GPU (the shared vLLM), 8 CPUs (the ailab
#   QOS caps cores at 8/GPU — do NOT raise without adding GPUs), 240G RAM,
#   walltime = SUM of all benchmarks (~8-12h for the full set). MLS + Theta are
#   the long tails; a fcs+ale+theta+ttt-only run (SKIP_MLS=1) fits in ~4-6h.
#   MLS's local-apptainer CPU pool shares these 8 cores with vLLM; that is the
#   same footprint the standalone MLS launcher uses on ailab.
#
# USAGE
#   sbatch --job-name=cc-all-<TAG> \
#     --export=ALL,MODEL_PATH=<MODEL_DIR>,TAG=<TAG> \
#     slurm/cc_eval_all_benchmarks.sh
#   SMOKE:
#     sbatch --job-name=cc-all-smoke \
#       --export=ALL,MODEL_PATH=<M>,TAG=smoke,VALIDATE=1,SKIP_MLS=1,BENCHMARKS="fcs ale theta ttt" \
#       slurm/cc_eval_all_benchmarks.sh
# =============================================================================
#SBATCH --job-name=cc-eval-all
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -uo pipefail   # NOTE: deliberately NOT -e. Benchmarks are fail-soft; a
                   # single benchmark's nonzero exit must never abort the job.

# ---- HARDCODED roots (BASH_SOURCE points at the slurm spool copy under sbatch) --
PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
THETA_ROOT="/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve"
OE_ROOT="$THETA_ROOT/openevolve_adapted"
cd "$PROJECT_ROOT"
mkdir -p logs

# ---- inputs ------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-9B}}"
TAG="${TAG:-$(basename "$MODEL_PATH")}"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "ERROR: MODEL_PATH=$MODEL_PATH has no config.json" >&2
  exit 1
fi
export MODEL_PATH TAG

# Which benchmarks to run (default all). Commas or spaces accepted.
BENCHMARKS="${BENCHMARKS:-fcs ale mls theta ttt}"
BENCHMARKS="${BENCHMARKS//,/ }"
want() { case " $BENCHMARKS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# fcs and ale share ONE process; running either runs both.
RUN_FCSALE=0; want fcs && RUN_FCSALE=1; want ale && RUN_FCSALE=1
RUN_MLS=0;    want mls && RUN_MLS=1
[ "${SKIP_MLS:-0}" = "1" ] && RUN_MLS=0
RUN_THETA=0;  want theta && RUN_THETA=1
RUN_TTT=0;    want ttt && RUN_TTT=1

ALL_BASE="${ALL_BASE:-$PROJECT_ROOT/outputs/cc_eval_all_${TAG}}"
SUMMARY_ALL="$ALL_BASE/summary_all.json"
LOGS_DIR="$ALL_BASE/bench_logs"
mkdir -p "$ALL_BASE" "$LOGS_DIR"

echo "[cc-all] TAG=$TAG MODEL=$MODEL_PATH"
echo "[cc-all] BENCHMARKS='$BENCHMARKS' -> fcs/ale=$RUN_FCSALE mls=$RUN_MLS theta=$RUN_THETA ttt=$RUN_TTT"
echo "[cc-all] output base: $ALL_BASE"

# ---- common offline / HF env -------------------------------------------------
export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TMPDIR="/tmp"
export OPENAI_API_KEY=EMPTY
export EVAL_RESEARCHER_YEAR="${EVAL_RESEARCHER_YEAR:-2026}"

# ---- ALE-Bench env (same as cc_eval_thinking_both) ---------------------------
export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

# ---- JOB-UNIQUE ports / cgroup / judge dirs (same scheme as thinking_both) ---
JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
SLOT=$(( (JOBU * 7) % 400 ))
export VLLM_PORT="${VLLM_PORT:-$(( 34000 + SLOT*20 ))}"
export PORT="${PORT:-$(( 21000 + SLOT*20 ))}"       # FrontierCS node judge API
export GJ_PORT="${GJ_PORT:-$(( 47000 + SLOT*20 ))}" # go-judge HTTP
export GJ_PARALLELISM="${GJ_PARALLELISM:-8}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export GJ_CGROUP_PREFIX="${GJ_CGROUP_PREFIX:-gojudge-${SLURM_JOB_ID:-$$}}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-ccall-${SLURM_JOB_ID:-manual}}"
export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"
export HOST=127.0.0.1

# ---- shared served-model name(s) ---------------------------------------------
# FCS/ALE/Theta use the SINGLE token CC_SERVED (word-splits to one arg). MLS needs
# bare <TAG> and vllm/<TAG>. Register all three; a request matches ANY of them.
CC_SERVED="cc-${TAG}-think"
export SERVED_MODEL_NAME="${CC_SERVED} ${TAG} vllm/${TAG}"

# ---- vLLM sizing (thinking, 32k budget — same as thinking_both) --------------
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
print(min(cap, 8900 + mt))
PY
  )"
fi
export MAX_MODEL_LEN
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[cc-all] vLLM=$VLLM_PORT judge node=$PORT go-judge=$GJ_PORT served='$SERVED_MODEL_NAME'"
echo "[cc-all] MAX_TOKENS=$MAX_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN"

# ============================================================================
# 1) Serve the model via vLLM ONCE (with MLS tool-call parser; harmless to rest)
# ============================================================================
PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  scripts/start_vllm_server.sh --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOGS_DIR/vllm.log" 2>&1 &
VLLM_PID="$!"
JUDGE_PID=""

cleanup() {
  [ -n "${JUDGE_PID:-}" ] && kill "$JUDGE_PID" >/dev/null 2>&1 || true
  kill "$VLLM_PID" >/dev/null 2>&1 || true
  [ -n "${JUDGE_PID:-}" ] && wait "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[cc-all] waiting for vLLM /health on ${VLLM_PORT} ..."
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "FATAL: vLLM exited early; tail of vllm.log:" >&2; tail -n 60 "$LOGS_DIR/vllm.log" >&2; exit 1
  fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
  || { echo "FATAL: vLLM never served /v1/models" >&2; tail -n 60 "$LOGS_DIR/vllm.log" >&2; exit 1; }
echo "[cc-all] vLLM ready on ${VLLM_PORT}. Registered models:"
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | sed 's/^/    /' || true

# ============================================================================
# 2) Start the FrontierCS go-judge + ALE apptainer ONCE (only if FCS/ALE run)
# ============================================================================
JUDGE_UP=0
if [ "$RUN_FCSALE" = "1" ]; then
  echo "[cc-all] starting FrontierCS judge (go-judge ${GJ_PORT} + node ${PORT}) ..."
  scripts/start_frontiercs_judge_hybrid.sh > "$LOGS_DIR/judge.log" 2>&1 &
  JUDGE_PID="$!"
  echo "[cc-all] waiting for go-judge /version ..."
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 && { JUDGE_UP=1; break; }
    sleep 0.5
    kill -0 "$JUDGE_PID" >/dev/null 2>&1 || { echo "[cc-all] judge launcher exited early" >&2; break; }
  done
  if [ "$JUDGE_UP" = 1 ]; then
    echo "[cc-all] waiting for node judge /health ..."
    NODE_UP=0
    for _ in $(seq 1 240); do
      curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { NODE_UP=1; break; }
      sleep 0.5
      kill -0 "$JUDGE_PID" >/dev/null 2>&1 || { echo "[cc-all] judge launcher exited early" >&2; break; }
    done
    [ "$NODE_UP" = 1 ] || JUDGE_UP=0
  fi
  # Final go-judge liveness re-check right before scoring.
  [ "$JUDGE_UP" = 1 ] && { curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 || JUDGE_UP=0; }
  if [ "$JUDGE_UP" = 1 ]; then
    echo "[cc-all] judge fully ready: go-judge ${GJ_PORT} + node ${PORT}"
  else
    echo "[cc-all] WARNING: FrontierCS judge did NOT come up; FCS will be recorded FAILED (ALE is judge-independent but the fcs+ale process still needs the judge URL — see below)." >&2
    tail -n 40 "$LOGS_DIR/judge.log" >&2 || true
  fi
fi

# Per-benchmark status file: each block writes "<name> <SUCCESS|FAILED>".
STATUS_FILE="$ALL_BASE/bench_status.txt"
: > "$STATUS_FILE"
record() { echo "$1 $2" >> "$STATUS_FILE"; }

VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

# ============================================================================
# 3) Run each benchmark FAIL-SOFT against the shared server
# ============================================================================

# ---- FCS + ALE (one process; reuse eval_base_model_qwen35_9b_vllm_request.sh) --
if [ "$RUN_FCSALE" = "1" ]; then
  FCSALE_BASE="$ALL_BASE/fcsale"
  FCSALE_SUMMARY="$FCSALE_BASE/summary.json"
  (
    set -uo pipefail
    export OUTPUT_DIR="$FCSALE_BASE/shard_0"
    export SUMMARY_JSON="$FCSALE_SUMMARY"
    export SAMPLES_JSONL="$OUTPUT_DIR/samples.jsonl"
    mkdir -p "$FCSALE_BASE" "$OUTPUT_DIR"
    # Decode 口径 (thinking-general, strip-think extract), same defaults as thinking_both.
    export ENABLE_THINKING="${ENABLE_THINKING:-1}"
    export TEMPERATURE="${TEMPERATURE:-1.0}" TOP_P="${TOP_P:-0.95}" TOP_K="${TOP_K:-20}"
    export MIN_P="${MIN_P:-0.0}" PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}" REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
    export CONCURRENCY="${CONCURRENCY:-64}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-2400}"
    export SOURCE="both" N_SAMPLES="${N_SAMPLES:-5}" NUM_SHARDS=1 SHARD_IDX=0
    export FRONTIERCS_PROMPT_SOURCE="${FRONTIERCS_PROMPT_SOURCE:-official}"
    export FRONTIERCS_SCORE_BACKEND="${FRONTIERCS_SCORE_BACKEND:-official}"
    export FRONTIERCS_ITERATIVE_ROUNDS="${FRONTIERCS_ITERATIVE_ROUNDS:-1}"
    export SAVE_TEXT=1 TEXT_PREVIEW_CHARS="${TEXT_PREVIEW_CHARS:-0}" SEED="${SEED:-42}"
    export MAX_ERRORS="${MAX_ERRORS:-12}"
    export FRONTIERCS_STRIP_THINK_EXTRACT="${FRONTIERCS_STRIP_THINK_EXTRACT:-1}"
    if [ "${VALIDATE:-0}" = "1" ]; then
      export FRONTIERCS_LIMIT="${FRONTIERCS_LIMIT:-4}" ALEBENCH_LIMIT="${ALEBENCH_LIMIT:-2}"
    fi
    # The scorer talks to the SHARED vLLM (CC_SERVED) + SHARED judge.
    VLLM_BASE_URL="$VLLM_BASE_URL" \
    SERVED_MODEL_NAME="$CC_SERVED" \
    FRONTIERCS_JUDGE_URL="$FRONTIERCS_JUDGE_URL" \
    RESUME="${RESUME:-1}" \
    bash scripts/eval_base_model_qwen35_9b_vllm_request.sh
  ) > "$LOGS_DIR/fcsale.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$FCSALE_SUMMARY" ]; then
    echo "[cc-all] FCS+ALE done (rc=0) -> $FCSALE_SUMMARY"; record fcsale SUCCESS
  else
    echo "[cc-all] FCS+ALE FAILED (rc=$rc). tail:" >&2; tail -n 40 "$LOGS_DIR/fcsale.log" >&2
    record fcsale FAILED
  fi
fi

# ---- MLS-Bench (local apptainer pool; reuse mlsbench_run_cpu_tasks.py) --------
if [ "$RUN_MLS" = "1" ]; then
  MLS_BASE="$ALL_BASE/mls"
  MLS_SUMMARY="$MLS_BASE/summary.json"
  (
    set -uo pipefail
    MLSBENCH_ROOT="${MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}"
    [ -d "$MLSBENCH_ROOT/src/mlsbench" ] || { echo "MLSBENCH_ROOT invalid: $MLSBENCH_ROOT" >&2; exit 3; }
    mkdir -p "$MLS_BASE"
    DATA_ROOT="${MLSBENCH_DATA_ROOT:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
    SAVE_PATH="$MLS_BASE/saves"; mkdir -p "$SAVE_PATH"
    GEN_CONFIG="$MLS_BASE/config_vllm_local_${SLURM_JOB_ID:-manual}.yaml"
    cat > "$GEN_CONFIG" <<YAML
# AUTO-GENERATED by cc_eval_all_benchmarks.sh — local Apptainer, NO slurm block.
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
    base_url: "${VLLM_BASE_URL}"
YAML
    export MLSBENCH_NO_PREBUILT=1 MLSBENCH_SCHEDULER_MANAGED=1 MLSBENCH_USE_REPLACE="${MLSBENCH_USE_REPLACE:-1}"
    AGENT_MODEL="vllm/${TAG}"
    MLSBENCH_PY="${MLSBENCH_PY:-/home/bl3615/miniconda3/bin/python}"
    [ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"
    EXTRA=()
    if [ -n "${SMOKE_TASK:-}" ]; then EXTRA+=(--tasks "$SMOKE_TASK"); fi
    if [ -n "${MLS_TASKS:-}" ]; then EXTRA+=(--tasks $MLS_TASKS); fi
    if [ -n "${MLS_LIMIT:-}" ]; then EXTRA+=(--limit "$MLS_LIMIT"); fi
    [ "${VALIDATE:-0}" = "1" ] && [ ${#EXTRA[@]} -eq 0 ] && EXTRA+=(--limit 1)
    MODEL="$AGENT_MODEL" MLSBENCH_ROOT="$MLSBENCH_ROOT" \
    "$MLSBENCH_PY" "$PROJECT_ROOT/scripts/mlsbench_run_cpu_tasks.py" \
      --config "$GEN_CONFIG" --model "$AGENT_MODEL" --root "$MLSBENCH_ROOT" \
      --out "$MLS_SUMMARY" --concurrency "${MLS_CONCURRENCY:-20}" \
      --timeout "${TASK_TIMEOUT:-5400}" --python "$MLSBENCH_PY" "${EXTRA[@]}"
  ) > "$LOGS_DIR/mls.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$MLS_SUMMARY" ]; then
    echo "[cc-all] MLS done (rc=0) -> $MLS_SUMMARY"; record mls SUCCESS
  else
    echo "[cc-all] MLS FAILED (rc=$rc). tail:" >&2; tail -n 40 "$LOGS_DIR/mls.log" >&2
    record mls FAILED
  fi
fi

# ---- Theta / TTT: shared OpenEvolve driver (reuse OpenEvolve engine internals) --
# The theta/ttt launchers each start their OWN vLLM; we can't reuse them as-is
# against a shared server. Instead we run the SAME OpenEvolve loop the launcher
# runs (cd OE_ROOT; python -m openevolve.cli ...) directly, pointing OPENAI_API_BASE
# at the shared server and rewriting the config's served name to CC_SERVED. This is
# byte-identical to the launcher's internals from line "cd OE_ROOT" onward; only the
# serve-vLLM prefix (which we already did once) is dropped.
run_openevolve_task() {
  local group="$1" task="$2" iters="$3" outbase="$4"
  # INITIAL/EVALUATOR are RELATIVE to $OE_ROOT (openevolve.cli is run after cd
  # "$OE_ROOT" below, exactly like the standalone launcher). But the config
  # existence checks + the sed source run while CWD is $PROJECT_ROOT, so they must
  # use ABSOLUTE $OE_ROOT paths -- else the -f test is evaluated in the wrong dir
  # and always falls through to the (nonexistent) smoke config.
  local INITIAL="examples/${task}/initial_programs/initial_program.py"
  local EVALUATOR="examples/${task}/evaluators/evaluator_modular.py"
  local FAITHFUL_SRC="$OE_ROOT/examples/${task}/configs/config_${task}_qwen35_faithful.yaml"
  local SMOKE_SRC="$OE_ROOT/examples/${task}/configs/config_${task}_qwen35_local_smoke.yaml"
  local CONFIG_SRC
  if [ "${CONFIG_VARIANT:-faithful}" = "smoke" ]; then CONFIG_SRC="$SMOKE_SRC"
  elif [ -f "$FAITHFUL_SRC" ]; then CONFIG_SRC="$FAITHFUL_SRC"
  else CONFIG_SRC="$SMOKE_SRC"; fi
  if [ ! -f "$CONFIG_SRC" ]; then echo "no config for $task ($CONFIG_SRC)" >&2; return 2; fi
  local OUT="$outbase/${task}"
  mkdir -p "$OUT"
  local CONFIG="$OUT/config_used.yaml"
  sed -e "s#api_base:.*#api_base: \"${VLLM_BASE_URL}\"#" \
      -e "s#\(- *name:\) *\"\?qwen35-9b\"\?#\1 \"${CC_SERVED}\"#" \
      "$CONFIG_SRC" > "$CONFIG"
  (
    cd "$OE_ROOT" || return 2
    export PYTHONPATH="$OE_ROOT:$THETA_ROOT:${PYTHONPATH:-}"
    export OPENAI_API_BASE="$VLLM_BASE_URL"
    python -m openevolve.cli "$INITIAL" "$EVALUATOR" \
      --config "$CONFIG" --output "$OUT" \
      --iterations "$iters" --random-seed "${SEED:-3407}" --log-level INFO
  ) || return $?
  python "$PROJECT_ROOT/scripts/parse_openevolve_best.py" \
    --run-dir "$OUT" --task "$task" --tag "${group}_${TAG}" --out "$OUT/summary.json"
}

# Theta: the 5 canonical tasks (per the campaign matrix). objective_value 口径.
if [ "$RUN_THETA" = "1" ]; then
  THETA_ITERS="${THETA_ITERATIONS:-12}"
  DEFAULT_THETA_TASKS="circle_packing_modular second_autocorr_inequality first_autocorr_inequality third_autocorr_inequality hadamard_matrix"
  T_TASKS="${THETA_TASKS:-$DEFAULT_THETA_TASKS}"
  if [ "${VALIDATE:-0}" = "1" ]; then
    T_TASKS="${THETA_TASKS:-circle_packing_modular}"
    THETA_ITERS="${THETA_ITERATIONS:-3}"
  fi
  THETA_BASE="$ALL_BASE/theta"
  for task in $T_TASKS; do
    (
      set -uo pipefail
      CONFIG_VARIANT="${CONFIG_VARIANT:-faithful}" \
        run_openevolve_task theta "$task" "$THETA_ITERS" "$THETA_BASE"
    ) > "$LOGS_DIR/theta_${task}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$THETA_BASE/${task}/summary.json" ]; then
      echo "[cc-all] Theta/$task done -> $THETA_BASE/${task}/summary.json"; record "theta:${task}" SUCCESS
    else
      echo "[cc-all] Theta/$task FAILED (rc=$rc). tail:" >&2; tail -n 25 "$LOGS_DIR/theta_${task}.log" >&2
      record "theta:${task}" FAILED
    fi
  done
fi

# TTT: AC1 (first_autocorr) + AC2 (second_autocorr) + circle ONLY (NOT third_autocorr).
if [ "$RUN_TTT" = "1" ]; then
  TTT_ITERS="${THETA_ITERATIONS:-12}"
  DEFAULT_TTT_TASKS="first_autocorr_inequality second_autocorr_inequality circle_packing_modular"
  TT_TASKS="${TTT_TASKS:-$DEFAULT_TTT_TASKS}"
  if [ "${VALIDATE:-0}" = "1" ]; then
    TT_TASKS="${TTT_TASKS:-second_autocorr_inequality}"
    TTT_ITERS="${THETA_ITERATIONS:-3}"
  fi
  TTT_BASE="$ALL_BASE/ttt"
  for task in $TT_TASKS; do
    case "$task" in
      first_autocorr_inequality|second_autocorr_inequality|circle_packing_modular) : ;;
      *) echo "[cc-all] TTT: refusing non-TTT task '$task' (valid: AC1/AC2/circle)"; record "ttt:${task}" FAILED; continue ;;
    esac
    (
      set -uo pipefail
      CONFIG_VARIANT="${CONFIG_VARIANT:-faithful}" \
        run_openevolve_task ttt "$task" "$TTT_ITERS" "$TTT_BASE"
    ) > "$LOGS_DIR/ttt_${task}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$TTT_BASE/${task}/summary.json" ]; then
      echo "[cc-all] TTT/$task done -> $TTT_BASE/${task}/summary.json"; record "ttt:${task}" SUCCESS
    else
      echo "[cc-all] TTT/$task FAILED (rc=$rc). tail:" >&2; tail -n 25 "$LOGS_DIR/ttt_${task}.log" >&2
      record "ttt:${task}" FAILED
    fi
  done
fi

# ============================================================================
# 4) Aggregate EVERYTHING into ONE summary_all.json
# ============================================================================
echo "[cc-all] aggregating -> $SUMMARY_ALL"
ALL_BASE="$ALL_BASE" TAG="$TAG" MODEL_PATH="$MODEL_PATH" SUMMARY_ALL="$SUMMARY_ALL" \
STATUS_FILE="$STATUS_FILE" python3 - <<'PY'
import json, os
from pathlib import Path

base = Path(os.environ["ALL_BASE"])
out = {
    "tag": os.environ["TAG"],
    "model_path": os.environ["MODEL_PATH"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "frontiercs": "NOT_RUN",
    "alebench": "NOT_RUN",
    "mls": "NOT_RUN",
    "theta": {},
    "ttt": {},
    "benchmarks_succeeded": [],
    "benchmarks_failed": [],
}

# per-benchmark run/status (written by the shell as "<name> <SUCCESS|FAILED>")
status = {}
sf = Path(os.environ.get("STATUS_FILE", ""))
if sf.is_file():
    for line in sf.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            status[parts[0]] = parts[1]

def getpath(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

# ---- FCS + ALE ----
# Read the summary REGARDLESS of the fcsale process exit code: FCS and ALE run in the
# same process, so an ALE infra failure (e.g. missing apptainer image -> >max-errors ->
# non-zero rc) must NOT discard a perfectly good FCS score. Decouple them per-metric.
st = status.get("fcsale")
if st is not None:
    fp = base / "fcsale" / "summary.json"
    if fp.is_file():
        d = json.loads(fp.read_text())
        fcs = getpath(d, "metrics", "frontiercs", "reward", "mean@5")
        ale = getpath(d, "metrics", "alebench", "performance", "mean@5")
        out["frontiercs"] = fcs if fcs is not None else "FAILED"
        # ALE all-samples-errored reports performance 0.0; treat 0.0 from a FAILED-rc run as infra-fail
        if ale is None or (st == "FAILED" and ale == 0.0):
            out["alebench"] = "FAILED"
        else:
            out["alebench"] = ale
    else:
        out["frontiercs"] = "FAILED"
        out["alebench"]   = "FAILED"

# ---- MLS ----
st = status.get("mls")
if st is not None:
    fp = base / "mls" / "summary.json"
    if st == "SUCCESS" and fp.is_file():
        d = json.loads(fp.read_text())
        out["mls"] = d.get("mean_score", "FAILED")
    else:
        out["mls"] = "FAILED"

# ---- Theta / TTT (per task; read best_objective_value NOT combined_score) ----
def collect(group):
    res = {}
    gdir = base / group
    for name, st in status.items():
        if not name.startswith(group + ":"):
            continue
        task = name.split(":", 1)[1]
        fp = gdir / task / "summary.json"
        if st == "SUCCESS" and fp.is_file():
            d = json.loads(fp.read_text())
            v = d.get("best_objective_value")
            res[task] = v if v is not None else "FAILED"
        else:
            res[task] = "FAILED"
    return res

out["theta"] = collect("theta")
out["ttt"]   = collect("ttt")

# ---- succeeded / failed rollup ----
def is_ok(v):
    return isinstance(v, (int, float))
for name in ("frontiercs", "alebench", "mls"):
    v = out[name]
    if v == "NOT_RUN":
        continue
    (out["benchmarks_succeeded"] if is_ok(v) else out["benchmarks_failed"]).append(name)
for group in ("theta", "ttt"):
    for task, v in out[group].items():
        label = f"{group}:{task}"
        (out["benchmarks_succeeded"] if is_ok(v) else out["benchmarks_failed"]).append(label)

Path(os.environ["SUMMARY_ALL"]).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo "[cc-all] DONE. summary_all -> $SUMMARY_ALL"
echo "===== summary_all.json ====="
cat "$SUMMARY_ALL" 2>/dev/null || echo "(no summary_all.json written)"

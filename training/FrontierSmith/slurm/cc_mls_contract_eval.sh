#!/usr/bin/env bash
# =============================================================================
# cc_mls_contract_eval.sh — run the MLS-Bench edit-CONTRACT A/B on TASK SCORE.
#
# One self-contained job: serve the model once on this node's GPU, then run
# (arm x rep x task) work items against it. Because the arms differ only by env
# vars passed to `mlsbench agent`, a single vLLM server serves every arm — the
# GPU is loaded once for the whole comparison, which is the only reason this is
# affordable while training holds the per-user GPU quota.
#
# Resumability: re-submitting the SAME command with the SAME OUT_DIR resumes.
# Per-task records are banked atomically; a cell is sealed only when all 20
# tasks are terminal, and `report` reads sealed cells only.
#
# Usage:
#   sbatch --job-name=cc-mls-contract \
#     --export=ALL,MODEL_PATH=<DIR>,TAG=<tag>,REPS=2,OUT_DIR=<dir> \
#     slurm/cc_mls_contract_eval.sh
#
# Key knobs (env):
#   TAG          model tag; also the prefix of every per-cell served name
#   REPS         replicates per arm            (default 2)
#   REP_START    first replicate index         (default 0) — extend a run by
#                re-submitting with REP_START=<old REPS> and the same OUT_DIR
#   ARMS         space-separated arm names     (default: all four)
#   ARMS_FILE    JSON arm spec; how a NEW contract joins without a code change
#   CONCURRENCY  parallel work items           (default 20)
#   MAX_ATTEMPTS retries for INFRASTRUCTURE failures (default 2). Raise it when
#                resuming after fixing an infra cause, so items that already
#                exhausted their attempts get another go.
#   PILOT_TASKS  restrict the panel (pilot only — a restricted panel must NEVER
#                be quoted as a 20-task mean)
# =============================================================================
# DEFAULTS TARGET gpu-ee, NOT ailab, on purpose. The 20 CPU tasks are CPU-bound
# between model calls (the GPU sits near 10-15% util), so throughput is set by
# cores per GPU -- and ailab hard-caps that at 8 and REJECTS any larger request.
# gpu-ee's A100 nodes have 128 cores, take 32+ cores per GPU, and bill against a
# different QOS, so this eval stops competing with training for the ailab GPU
# quota. Request a WHOLE a100 (gpu:a100:1) so Slurm cannot hand back a MIG slice.
# Override with sbatch flags to run anywhere else.
#SBATCH --job-name=cc-mls-contract
#SBATCH --partition=gpu-ee
#SBATCH --account=chij
#SBATCH --qos=della-gpuee
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Under sbatch $BASH_SOURCE points at the spool copy, so hardcode the root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"
mkdir -p logs

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
TAG="${TAG:-$(basename "$MODEL_PATH")}"
[ -e "$MODEL_PATH/config.json" ] || { echo "ERROR: no config.json in $MODEL_PATH" >&2; exit 1; }

MLSBENCH_ROOT="${MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev}"
[ -d "$MLSBENCH_ROOT/src/mlsbench" ] || { echo "ERROR: $MLSBENCH_ROOT is not an MLS-Bench checkout" >&2; exit 1; }

REPS="${REPS:-2}"
REP_START="${REP_START:-0}"
ARMS="${ARMS:-linerange replace_strict replace_fx replace_fx_view}"
ARMS_FILE="${ARMS_FILE:-}"
# CONCURRENCY is bounded by GPU KV CACHE, not by cores. Measured on this A100
# with max_model_len 40960: at 30 concurrent agents the KV cache sits at 95-98%
# and aggregate generation collapses to 150-470 tok/s -- 5-15 tok/s per request
# -- so a single thinking response (up to ~14k tokens) blows past the OpenAI
# client's 1200 s timeout and the episode dies with APITimeoutError. Total
# throughput is ~30 episodes/GPU-hour either way, since it is set by generation
# throughput; raising concurrency past the KV budget buys nothing and costs
# timeouts plus the GPU spent on retries. 12 leaves ~2x headroom on an 80 GB
# A100; an H200 (141 GB) sustains ~20.
CONCURRENCY="${CONCURRENCY:-12}"
# Per-item wall-clock cap. 5400 s (the value the single-run MLS eval uses) is
# too short HERE: that runner gives a task the whole node, whereas this one has
# CONCURRENCY agents sharing one GPU, so a task that takes ~2300 s alone can
# take >5400 s under load. In the pilot that truncated 6 of 160 items -- all on
# the three slowest tasks (causal-discovery-discrete, optimization-evolution-
# strategy, causal-observational-linear-non-gaussian) -- and a truncation is not
# neutral: it lands on the runs that generate the most, so it biases against
# whichever arm is doing the most work.
TASK_TIMEOUT="${TASK_TIMEOUT:-14400}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/cc_mls_contract_${TAG}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
# Host-side BLAS/OpenMP pinning: MLS-Bench parses test output in a thread pool
# and some parsers enter LAPACK; concurrent LAPACK entry SIGSEGVs the agent
# process on this 64-core node. Not in mlsbench's PASSTHROUGH_ENV_VARS, so
# in-container compute keeps full threads.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export EVAL_RESEARCHER_YEAR="${EVAL_RESEARCHER_YEAR:-2026}"
export MLSBENCH_NO_PREBUILT=1
export MLSBENCH_SCHEDULER_MANAGED=1
# The arm spec owns the contract knobs. Clear anything inherited so a stale
# value from the caller's environment cannot silently redefine an arm.
unset MLSBENCH_USE_REPLACE MLSBENCH_STRICT_STR_REPLACE MLSBENCH_VIEW_TOOL || true

JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
export VLLM_PORT="${VLLM_PORT:-$(( 34000 + JOBU ))}"
export HOST=127.0.0.1
export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export DTYPE="${DTYPE:-bfloat16}"

MLSBENCH_PY="${MLSBENCH_PY:-/home/bl3615/miniconda3/bin/python}"
[ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"

RUNNER="$PROJECT_ROOT/scripts/mlsbench_contract_eval.py"

ARMS_ARG=()
[ -n "$ARMS_FILE" ] && ARMS_ARG+=(--arms-file "$ARMS_FILE")

# Every (arm, rep) cell needs its OWN leaderboard identity: `mlsbench score`
# groups tasks/<task>/leaderboard.csv rows by model string and returns one row
# per model across the file's whole history, so replicates sharing a model
# string would re-read each other's rows. Register all of them (bare + vllm/
# prefixed) as vLLM aliases.
SERVED_MODEL_NAME="$("$MLSBENCH_PY" "$RUNNER" served-names \
  --tag "$TAG" --arms $ARMS --reps "$REPS" --rep-start "$REP_START" "${ARMS_ARG[@]}")"
export SERVED_MODEL_NAME
echo "[contract] TAG=$TAG arms=($ARMS) reps=$REPS from $REP_START"
echo "[contract] serving $(wc -w <<<"$SERVED_MODEL_NAME") aliases on port $VLLM_PORT"
echo "[contract] OUT_DIR=$OUT_DIR MLSBENCH_ROOT=$MLSBENCH_ROOT"

# ----- MLS-Bench config (local apptainer, NO slurm block) --------------------
DATA_ROOT="${MLSBENCH_DATA_ROOT:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
SAVE_PATH="${MLSBENCH_SAVE_PATH:-$OUT_DIR/saves}"
mkdir -p "$SAVE_PATH"
GEN_CONFIG="$OUT_DIR/config_${SLURM_JOB_ID:-manual}.yaml"
cat > "$GEN_CONFIG" <<YAML
# AUTO-GENERATED by cc_mls_contract_eval.sh — local Apptainer, NO slurm block.
# seeds is FIXED across every arm and replicate: the task data is common random
# numbers, so the only thing that differs between arms is the edit contract.
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
echo "[contract] config: $GEN_CONFIG"

# ----- start vLLM ------------------------------------------------------------
PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" MODEL_PATH="$MODEL_PATH" \
  scripts/start_vllm_server.sh --enable-auto-tool-choice --tool-call-parser hermes \
  > "$OUT_DIR/vllm_${SLURM_JOB_ID:-manual}.log" 2>&1 &
VLLM_PID="$!"
KEEPALIVE_PID=""
cleanup() {
  [ -n "$KEEPALIVE_PID" ] && kill "$KEEPALIVE_PID" >/dev/null 2>&1 || true
  kill "$VLLM_PID" >/dev/null 2>&1 || true
  wait "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[contract] waiting for vLLM /health on ${VLLM_PORT} ..."
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
  kill -0 "$VLLM_PID" >/dev/null 2>&1 || {
    echo "vLLM exited early; tail:" >&2; tail -60 "$OUT_DIR/vllm_${SLURM_JOB_ID:-manual}.log" >&2; exit 1; }
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
  || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
FIRST_NAME="$(awk '{print $1}' <<<"$SERVED_MODEL_NAME")"
echo "[contract] vLLM ready; registered $(curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" \
  | tr ',' '\n' | grep -c '"id"') model ids"

# GPU keep-alive: the tasks are CPU-bound for long stretches and the cluster
# auto-cancels a GPU job idle at 0% util for 90 min.
( while true; do sleep 480; \
    curl -s -m 30 "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${FIRST_NAME}\",\"prompt\":\"ping\",\"max_tokens\":8}" \
      >/dev/null 2>&1 || true; done ) &
KEEPALIVE_PID="$!"

# ----- run the work queue ----------------------------------------------------
TASK_ARG=()
[ -n "${PILOT_TASKS:-}" ] && TASK_ARG+=(--tasks $PILOT_TASKS)

MLSBENCH_ROOT="$MLSBENCH_ROOT" \
"$MLSBENCH_PY" "$RUNNER" run \
  --out "$OUT_DIR" \
  --tag "$TAG" \
  --root "$MLSBENCH_ROOT" \
  --config "$GEN_CONFIG" \
  --arms $ARMS "${ARMS_ARG[@]}" \
  --reps "$REPS" --rep-start "$REP_START" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TASK_TIMEOUT" \
  --max-attempts "${MAX_ATTEMPTS:-2}" \
  --python "$MLSBENCH_PY" \
  --workspace-root "${CONTRACT_WS_ROOT:-$OUT_DIR/ws}" \
  "${TASK_ARG[@]}"

echo "===== report ====="
"$MLSBENCH_PY" "$RUNNER" report --out "$OUT_DIR" || true
echo "[contract] DONE -> $OUT_DIR"

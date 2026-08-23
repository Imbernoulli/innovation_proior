#!/usr/bin/env bash
# =============================================================================
# ONE reusable, trustworthy thinking-mode eval for FrontierCS + ALE-Bench.
#
# Parameterized by MODEL_DIR (=MODEL_PATH) and TAG. Single 1-GPU job, single
# shard over ALL problems (no multi-shard co-location). Self-contained:
#   1. Starts the offline FrontierCS judge (go-judge + apptainer node server)
#      with JOB-UNIQUE ports, cgroup-prefix, AND go-judge work dir.
#   2. Health-gates scoring on BOTH go-judge /version AND node /health, then
#      keeps a background liveness monitor running. If go-judge ever dies the
#      job aborts loudly instead of scoring against a dead judge (the bug that
#      silently produced FrontierCS=0 across whole evals).
#   3. Serves the model via vLLM and generates in THINKING mode at a
#      model-appropriate budget (Qwen3-8B / Qwen3.5-9B -> 32K thinking, 40960 ctx).
#   4. SAVE_TEXT=1 so generations are persisted and re-scorable.
#   5. Scores FrontierCS with the FIXED official scorer (infra failures RAISE)
#      AND runs ALE-Bench; writes summary.json with BOTH metrics.
#
# Usage (single command for any model):
#   sbatch --job-name=cc-eval-<TAG> \
#     --export=ALL,MODEL_PATH=<MODEL_DIR>,TAG=<TAG> \
#     slurm/cc_eval_thinking_both_ailab.sh
#
# Short validation (<=1h, 1 GPU): add VALIDATE=1 (limits to a few problems).
# =============================================================================
#SBATCH --job-name=cc-eval-thinking
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail

# vLLM penalty fast path (2026-08-15). The eval decode profile uses
# presence_penalty=1.5, which puts vLLM's sampler on a CPU path that rebuilds a
# [batch, max_len] int64 tensor EVERY decode step (measured 162 ms/step at
# B=128 L=32768, ~70% of the step, GPUs idling at 0-37%). The replacement in
# scripts/vllm_penalty_fastpath.py appends only new tokens; output is
# bit-identical, so scores are unaffected. Installed in the venv and inert
# unless this is 1 -- see scripts/apply_vllm_penalty_fastpath.sh.
export FS_VLLM_PENALTY_FASTPATH="${FS_VLLM_PENALTY_FASTPATH:-1}"


# PROJECT_ROOT is a HARDCODED absolute path (=FrontierSmith). Do NOT derive it
# from ${BASH_SOURCE[0]}: under sbatch the script runs from Slurm's spool dir
# (/var/spool/slurmd/.../slurm_script), so BASH_SOURCE-based resolution pointed
# at a non-writable spool parent -> `mkdir -p logs` => "Permission denied" and the
# eval died in ~3s. The theta eval works precisely because it hardcodes this path.
PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
cd "$PROJECT_ROOT"
mkdir -p logs

# ----- inputs -----------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-9B}}"
TAG="${TAG:-$(basename "$MODEL_PATH")}"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "ERROR: MODEL_PATH=$MODEL_PATH has no config.json" >&2
  exit 1
fi
export MODEL_PATH

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/cc_eval_${TAG}_thinking_32k_both_vllm}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/shard_0}"
export SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_BASE/summary.json}"
export SAMPLES_JSONL="${SAMPLES_JSONL:-$OUTPUT_DIR/samples.jsonl}"
mkdir -p "$OUTPUT_BASE" "$OUTPUT_DIR"

# ----- log noise filter -------------------------------------------------------
# Drop the noisy real_accelerator/ds_accelerator import banners (and similar)
# without losing real errors. Applied to this script's stdout/stderr.
exec > >(grep -avE 'real_accelerator|ds_accelerator|Setting ds_accelerator') \
    2> >(grep -avE 'real_accelerator|ds_accelerator|Setting ds_accelerator' >&2)

# ----- env --------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TMPDIR="/tmp"

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

# ----- JOB-UNIQUE judge ports / cgroup / work dir -----------------------------
# Unique high ports avoid collisions; unique cgroup-prefix avoids the fixed
# gojudge.scope clash; unique go-judge -dir (set inside the starter from
# GJ_CGROUP_PREFIX) avoids the /dev/shm/go-judge collision that silently killed
# go-judge on co-located nodes.
JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
# STRIDE ports by 20 + decorrelate consecutive job ids: vLLM scans ~5 ports up from VLLM_PORT for its
# internal distributed init, so adjacent ports on a co-located node collide (EADDRINUSE). 400 slots x 20.
SLOT=$(( (JOBU * 7) % 400 ))
export PORT="${PORT:-$(( 21000 + SLOT*20 ))}"          # node judge API
export GJ_PORT="${GJ_PORT:-$(( 47000 + SLOT*20 ))}"    # go-judge HTTP
export VLLM_PORT="${VLLM_PORT:-$(( 34000 + SLOT*20 ))}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-8}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export GJ_CGROUP_PREFIX="${GJ_CGROUP_PREFIX:-gojudge-${SLURM_JOB_ID:-$$}}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-cceval-${SLURM_JOB_ID:-manual}}"
# Drive the scorer's judge URL to the ACTUAL node port (fixes 8082-vs-5050).
export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"

echo "[cc_eval] TAG=$TAG MODEL=$MODEL_PATH"
echo "[cc_eval] judge node PORT=$PORT go-judge GJ_PORT=$GJ_PORT vLLM=$VLLM_PORT cgroup=$GJ_CGROUP_PREFIX"
echo "[cc_eval] FRONTIERCS_JUDGE_URL=$FRONTIERCS_JUDGE_URL output=$SUMMARY_JSON"

# ----- model-appropriate thinking budget --------------------------------------
# Paper口径: 32K thinking budget (MAX_TOKENS=32768) for both Qwen3-8B and
# Qwen3.5-9B, for comparability. MAX_MODEL_LEN is sized so the longest prompt
# (FrontierCS statement ~8.8K tok) + 32768 output fits without truncation, but
# never exceeds the model's own context cap (Qwen3-8B is hard-capped at 40960).
export TP="${TP:-1}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
if [ -z "${MAX_MODEL_LEN:-}" ]; then
  MAX_MODEL_LEN="$(
    MODEL_PATH="$MODEL_PATH" MAX_TOKENS="$MAX_TOKENS" python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
# Model context cap (qwen3 exposes max_position_embeddings; qwen3_5 omits it and
# supports very long context, so fall back to a high cap there).
cap = cfg.get("max_position_embeddings") or 262144
mt = int(os.environ["MAX_TOKENS"])
# Want longest prompt (~8.8K) + full output to fit; cap at the model context.
want = 8900 + mt          # ~41668
print(min(cap, want))
PY
  )"
fi
export MAX_MODEL_LEN
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-cc-${TAG}-think}"
export HOST=127.0.0.1

# ----- sampling profile -------------------------------------------------------
# Two profiles:
#
#   default ("qwen-thinking"): Qwen3 thinking-general 口径 (temp 1.0 / top_p 0.95 /
#     top_k 20 / min_p 0 / presence_penalty 1.5). This is a Qwen-tuned decode for
#     LOCAL open-weights thinking models and is what our historical runs used.
#
#   FRONTIERCS_FAITHFUL_SAMPLING=1 ("official"): mirror the OFFICIAL Frontier-CS
#     closed-API generator (algorithmic/scripts/generate_solutions.py ->
#     gen/llm_interface.py). That pipeline sends the chat request with NO
#     temperature, NO top_p, and NO presence/frequency penalty -- it ONLY sets
#     reasoning_effort="high" for reasoning models and otherwise takes each
#     provider's server-side defaults. vLLM's server default is temp=1.0 with
#     top_p=1.0 and no penalties, so the faithful profile drops presence_penalty
#     entirely, uses temp 1.0 / top_p 1.0, and leaves top_k/min_p unset. Use this
#     to compare against the leaderboard numbers on equal footing; keep the
#     default for the Qwen-native runs. (presence_penalty=1.5 in particular is a
#     Qwen-specific anti-repetition knob the official pipeline never applies and
#     it measurably shifts scores, so it must be OFF for a faithful comparison.)
export ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [ "${FRONTIERCS_FAITHFUL_SAMPLING:-0}" = "1" ]; then
  export TEMPERATURE="${TEMPERATURE:-1.0}"
  export TOP_P="${TOP_P:-1.0}"
  export TOP_K="${TOP_K:-}"          # unset -> not sent (server default)
  export MIN_P="${MIN_P:-}"          # unset -> not sent (server default)
  export PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"   # unset -> not sent (official sends none)
  export REPETITION_PENALTY="${REPETITION_PENALTY:-}"
  echo "[cc_eval] FRONTIERCS_FAITHFUL_SAMPLING=1: official decode (temp=$TEMPERATURE top_p=$TOP_P, no presence_penalty/top_k/min_p)"
else
  export TEMPERATURE="${TEMPERATURE:-1.0}"
  export TOP_P="${TOP_P:-0.95}"
  export TOP_K="${TOP_K:-20}"
  export MIN_P="${MIN_P:-0.0}"
  export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
  export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
fi
export CONCURRENCY="${CONCURRENCY:-64}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-2400}"

export SOURCE="${SOURCE:-both}"
export N_SAMPLES="${N_SAMPLES:-5}"
export NUM_SHARDS="${NUM_SHARDS:-1}"
export SHARD_IDX="${SHARD_IDX:-0}"
export FRONTIERCS_PROMPT_SOURCE="${FRONTIERCS_PROMPT_SOURCE:-official}"
export FRONTIERCS_SCORE_BACKEND="${FRONTIERCS_SCORE_BACKEND:-official}"
# PROTOCOL: 1 = leaderboard-faithful SINGLE-SHOT (generate->extract->judge), the
# native Frontier-CS algorithmic protocol behind Score@1/Avg@5/Score@5 (default,
# unchanged). >1 = optional AGENTIC iterative refinement for the frontiercs track:
# feed the judge score back each round and keep the best round, mirroring the
# official Harbor agent adapter (README "records the higher of the final solution
# score and the best iterative submission"). Note: >1 makes numbers NON-comparable
# to the single-shot leaderboard.
export FRONTIERCS_ITERATIVE_ROUNDS="${FRONTIERCS_ITERATIVE_ROUNDS:-1}"
export SAVE_TEXT="${SAVE_TEXT:-1}"          # always persist generations (re-scorable)
export TEXT_PREVIEW_CHARS="${TEXT_PREVIEW_CHARS:-0}"
export SEED="${SEED:-42}"
# A handful of heavy FrontierCS problems can legitimately exceed the judge's
# per-submission evaluation timeout (status=timeout, ~1000s). Those are recorded
# LOUDLY as `error` per sample (never silent 0) and conservatively count as 0 in
# the aggregate -- but they should not hard-FAIL the whole job and block writing
# summary.json. Tolerate a small number; a large count still aborts (real trouble).
# This is NOT the judge-down case: ECONNREFUSED / dead-judge still abort via the
# health gates above before any scoring starts.
export MAX_ERRORS="${MAX_ERRORS:-12}"

# Short validation mode: cap problems so a job finishes in <=1h on 1 GPU.
if [ "${VALIDATE:-0}" = "1" ]; then
  export FRONTIERCS_LIMIT="${FRONTIERCS_LIMIT:-6}"
  export ALEBENCH_LIMIT="${ALEBENCH_LIMIT:-3}"
  echo "[cc_eval] VALIDATE mode: FRONTIERCS_LIMIT=$FRONTIERCS_LIMIT ALEBENCH_LIMIT=$ALEBENCH_LIMIT"
fi

# ----- start vLLM FIRST -------------------------------------------------------
# Ordering matters: vLLM's model load + CUDA-graph capture is a large, transient
# memory spike. go-judge (started later) keeps a small RAM footprint but was
# historically OOM-killed (SIGKILL) when it competed with that spike on a
# co-located node -- it bound its port, served /version, then died before the
# first /run, silently zeroing the whole eval. Bring vLLM fully up first, THEN
# start the judge into the now-stable memory state.
PORT="$VLLM_PORT" scripts/start_vllm_server.sh &
VLLM_PID="$!"
JUDGE_PID=""

cleanup() {
  [ -n "${JUDGE_PID:-}" ] && kill "$JUDGE_PID" >/dev/null 2>&1 || true
  kill "$VLLM_PID" >/dev/null 2>&1 || true
  [ -n "${JUDGE_PID:-}" ] && wait "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$VLLM_PID" >/dev/null 2>&1 || true
  # Remove this job's judge workdir. It holds ~9k files (the unpacked problem
  # data + logs) and was previously LEAKED on every run: 1209 stale dirs had
  # accumulated, ~9M inodes, and the shared CHIJ fileset hit 99.8M of its 100M
  # inode cap -- which would have failed every job in the campaign, training
  # checkpoints included. The dir is per-job (RUNTIME_DIR carries $SLURM_JOB_ID)
  # and fully rebuildable, so deleting it on exit is safe. Guard on the expected
  # prefix so a caller-supplied RUNTIME_DIR can never make this rm something else.
  case "${RUNTIME_DIR:-}" in
    "$PROJECT_ROOT"/.cache/frontiercs-judge-*)
      rm -rf "$RUNTIME_DIR" >/dev/null 2>&1 || true ;;
  esac
  rm -rf "${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[cc_eval] waiting for vLLM /health ..."
for _ in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then echo "vLLM server exited early" >&2; exit 1; fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
  || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
echo "[cc_eval] vLLM ready on ${VLLM_PORT}"

# ----- start judge AFTER vLLM is stable ---------------------------------------
echo "[cc_eval] starting FrontierCS judge (go-judge ${GJ_PORT} + node ${PORT}) ..."
scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

# ----- gate on BOTH go-judge /version AND node /health ------------------------
echo "[cc_eval] waiting for go-judge /version ..."
gj_ready=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1; then gj_ready=1; break; fi
  sleep 0.5
  if ! kill -0 "$JUDGE_PID" >/dev/null 2>&1; then echo "judge launcher exited early" >&2; exit 1; fi
done
[ "$gj_ready" = 1 ] || { echo "ERROR: go-judge never came up on ${GJ_PORT}" >&2; exit 1; }

echo "[cc_eval] waiting for node judge /health ..."
node_ready=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then node_ready=1; break; fi
  sleep 0.5
  if ! kill -0 "$JUDGE_PID" >/dev/null 2>&1; then echo "judge launcher exited early" >&2; exit 1; fi
done
[ "$node_ready" = 1 ] || { echo "ERROR: node judge never healthy on ${PORT}" >&2; exit 1; }
echo -n "[cc_eval] judge /health: "; curl -fsS "http://127.0.0.1:${PORT}/health"; echo

# Final go-judge liveness check right before scoring begins (it is the one that
# historically dies). If it died, abort -- NEVER score against a dead judge.
curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 \
  || { echo "ERROR: go-judge died after node came up (the classic silent-0 bug)" >&2; exit 1; }
echo "[cc_eval] judge fully ready: go-judge ${GJ_PORT} + node ${PORT}"

# ----- run eval (fixed scorer: FCS infra failures RAISE, recorded as error) ---
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
RESUME="${RESUME:-1}" \
bash scripts/eval_base_model_qwen35_9b_vllm_request.sh "$@"

echo "[cc_eval] DONE. summary -> $SUMMARY_JSON"

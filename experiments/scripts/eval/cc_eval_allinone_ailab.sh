#!/usr/bin/env bash
# cc_eval_allinone_ailab.sh -- vLLM serve AND the eval client inside ONE ailab job.
#
# Why: the split serve/client design fought the scheduler on every axis and lost.
#   - ailab refuses CPU-only jobs outright, so a client could never go there;
#   - gpu-ee is owned by another user whose whole queue outranks ours (nodes go
#     MIXED+PLANNED), so waves stalled there for hours;
#   - the `cpu` partition is 15k/15.6k cores allocated, so a client waits hours;
#   - and whenever the two halves did not overlap, the serve hit SERVE_GRACE_SEC and
#     died, stranding the client on "waiting for backend ... / 14400s".
# One job removes all four failure modes: the backend is up exactly when the client
# needs it, on the same node, over loopback.
#
# It also unlocks the research track, which the split design could never run: the
# FrontierCS research evaluator raises
#   RuntimeError("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
# so the client itself needs a real GPU -- impossible on `cpu`, natural here.
#
# ALWAYS ASK FOR gpu:1, NEVER gpu:2. The first version asked for two cards so the
# research evaluator could have its own, but the second card then sat at 0% and the
# cluster's idle-GPU sweep killed the jobs: 13356824 / 13356847 / 13356848 were all
# "CANCELLED by 123" at the same instant, 2026-09-02T17:45:05, after 1h32m-1h39m.
# One card, shared: vLLM holds it (so utilisation is never zero) and the research
# evaluator runs its kernels on the same device -- which only fits if vLLM leaves room,
# hence GPU_MEMORY_UTILIZATION=0.55 for SOURCE=research (0.92 is fine for `both`,
# whose client never touches CUDA).
#
# Usage (SOURCE: both | research | frontiercs | alebench):
#   sbatch --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G \
#     --time=08:00:00 --job-name=ev-<TAG>-<src><shard> \
#     --export=ALL,MODEL=<dir>,TAG=<tag>,SOURCE=both,NUM_SHARDS=2,SHARD_IDX=0 \
#     cc_eval_allinone_ailab.sh
set -uo pipefail

D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
PROJECT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$PROJECT_ROOT"

: "${MODEL:?set MODEL}" "${TAG:?set TAG}"
export SOURCE="${SOURCE:-both}"
export NUM_SHARDS="${NUM_SHARDS:-1}"
export SHARD_IDX="${SHARD_IDX:-0}"
export MODEL_TAG="${MODEL_TAG:-$TAG}"

# --- zy7019 env (bl3615's home is unreadable; everything must resolve into our tree) ---
export VLLM_VENV="${VLLM_VENV:-$D/envs/vllm023}"
export PATH="$D/envs/client/bin:$PATH"
export HF_HOME="${HF_HOME:-$D/.hf}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$D/.cache/vllm}"
export ALE_BENCH_CACHE="${ALE_BENCH_CACHE:-$D/.cache/ale-bench}"
export ALE_BENCH_CACHE_DIR="${ALE_BENCH_CACHE_DIR:-$D/.cache/ale-bench}"
export MLSBENCH_ROOT="${MLSBENCH_ROOT:-$D/mlsroot}"
export EVAL_RESEARCHER_YEAR="${EVAL_RESEARCHER_YEAR:-2026}"
export EVAL_SYS_PROMPT_MODE="${EVAL_SYS_PROMPT_MODE:-short}"
export MLSBENCH_SYS_PREFIX="${MLSBENCH_SYS_PREFIX:-It is now year 2026.}"

# --- research: julia/pysr must write into OUR tree, not bohan's ---------------
# The research evaluators run under bohan's research_overlay interpreter, whose
# defaults (frontiercs_research_cpu_eval.py:437, DEFAULT_OVERLAY) put the juliapkg
# project in HIS tree. zy7019 cannot write it, so every symbolic_regression sample
# came back as an infra error rather than a score:
#   ResearchInfraError("symbolic_regression/<prob>: evaluator produced no result
#     (rc=1): [Errno 13] Permission denied:
#     '/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_env/lock.pid'")
# That was 25-50 rows PER ARM (50 in the base9b_v2c anchor) -- the single largest
# error class on the research track, and it is pure infrastructure, not the model.
# cc_eval_all_selfc.sh already redirected these; the all-in-one path did not, which
# is why its research runs kept the same 25-50 hole. Same override, same copy.
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$D/envs/research_julia/julia_depot}"
export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-$D/envs/research_julia/julia_env}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"

# --- one card, shared: vLLM keeps it busy so the idle-GPU sweep never sees 0% ---
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 1)
[ "$NGPU" -ge 2 ] && echo "[allinone] WARNING: $NGPU GPUs allocated; the idle one will get this job swept. Submit with --gres=gpu:1." >&2
echo "[allinone] TAG=$TAG src=$SOURCE shard=$SHARD_IDX/$NUM_SHARDS gpus=$NGPU"

export TP="${TP:-1}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FS_VLLM_PENALTY_FASTPATH=1
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
# research runs CUDA kernels in-process on the same card, so vLLM must leave it room.
if [ "$SOURCE" = "research" ]; then _gmu_default=0.55; else _gmu_default=0.92; fi
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$_gmu_default}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"

VLLM_PORT="${VLLM_PORT:-$(python3 -c '
import socket
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
export VLLM_PORT

cleanup() { [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[allinone] starting vLLM on 127.0.0.1:$VLLM_PORT"
HOST=127.0.0.1 PORT="$VLLM_PORT" \
  MODEL_PATH="$MODEL" SERVED_MODEL_NAME="$TAG" \
  bash scripts/start_vllm_server.sh &
VLLM_PID=$!

ready=0
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[allinone] vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "[allinone] vLLM never became ready" >&2; exit 1; }
echo "[allinone] vLLM ready after $SECONDS s"

# VLLM_BASE_URL set => the client skips the pool registry and its 4 h POOL_WAIT entirely,
# which is the whole point: no registry, no dependency, no race.
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export SERVED_MODEL_NAME="$TAG"

# --- keep-alive: the idle-GPU sweep kills these jobs mid-run ------------------
# Holding the card is NOT enough -- the sweep looks at *utilisation*. During the
# ALE-Bench phase the client spends long stretches compiling and running C++ under
# go-judge without querying the model, so the GPU sits at 0% and the sweeper
# (uid 123 = the slurm daemon, firing on :00/:15/:30/:45) cancels the job:
#   13356824 / 13359987 (at SAMPLE 524/530) / 13367145 / 13367147 / 13373951
# were all "CANCELLED by 123", always the shard-0 job, always on a quarter hour.
# NOTE: sacct's Reason column reports QOSMaxGRESPerUser for 13359987; that is the
# stale last-pending reason, not the cancel cause -- the End timestamps line up with
# the sweeper, not with any submission of ours.
# One 1-token completion a minute is enough to keep utilisation off the floor. It
# goes to the same server on a throwaway request and never touches the sample records.
( while sleep 60; do
    curl -fsS -m 20 -X POST "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$TAG\",\"prompt\":\"ping\",\"max_tokens\":1,\"temperature\":0}" \
      >/dev/null 2>&1 || true
  done ) &
KEEPALIVE_PID=$!
cleanup() {
  [ -n "${KEEPALIVE_PID:-}" ] && kill "$KEEPALIVE_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" >/dev/null 2>&1 || true
}
echo "[allinone] keep-alive pinger started (pid $KEEPALIVE_PID, 1 tok/60 s)"

bash "$D/slurm_overlay/cc_eval_cpu_client_pinned.sh"
rc=$?
echo "[allinone] client exited rc=$rc after $SECONDS s"
exit $rc

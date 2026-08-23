#!/usr/bin/env bash
# Serve one vLLM backend for the eval POOL and register it so CPU-only eval jobs
# can find it.
#
# WHY: every eval job used to start its own vLLM on its own GPU node. That pays a
# ~5 min model load per job, and pins scoring to the same node -- on ailab the
# ratio is enforced at 8 cores/GPU, so a job that wants 16 cores must burn 2 GPUs
# purely to buy cores. Splitting serving from scoring lets a few GPUs feed many
# cheap CPU jobs.
#
# ROUTING: verified by experiment (jobs 12515903/12515916) that compute nodes can
# open TCP connections to each other directly -- a listener on della-r3c1n11:47311
# was reachable from another compute node. So NO login-node proxy is needed, and
# none is wanted: it would be a single point of failure and a bandwidth choke for
# 32k-token responses. The client connects straight to <node>:<port>.
#
# The one thing that had to change: vLLM defaulted to --host 127.0.0.1 (loopback
# only, unreachable from anywhere else). We bind 0.0.0.0 here.
#
# DISCOVERY: a registry file on shared GPFS. No daemon, no single point of
# failure; a client just reads the directory. Each backend owns exactly one file,
# writes it atomically, refreshes an mtime heartbeat, and removes it on exit, so a
# crashed backend ages out instead of being handed to clients forever.
#
# Usage:
#   MODEL=/path/to/hf_dir TAG=my-model scripts/vllm_pool_serve.sh
# Typically via sbatch, e.g.
#   sbatch --gres=gpu:1 -c 8 --mem=110G --export=ALL,MODEL=...,TAG=... \
#          --wrap 'bash scripts/vllm_pool_serve.sh'
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"

: "${MODEL:?set MODEL to an HF model dir}"
TAG="${TAG:-$(basename "$MODEL")}"
REGISTRY="${VLLM_POOL_REGISTRY:-$FS/.cache/vllm_pool}"
mkdir -p "$REGISTRY"

# Port: derive from the job id so two backends on one node cannot collide, and
# stay well above the ephemeral range.
PORT="${VLLM_PORT:-$(( 39000 + (${SLURM_JOB_ID:-$$} % 2000) ))}"
HOSTNAME_FQ="$(hostname -s)"
ENTRY="$REGISTRY/${TAG}__${HOSTNAME_FQ}__${PORT}.json"

cleanup() {
  rm -f "$ENTRY" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && wait "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Optimisations, all on for a pool backend. Rationale per knob:
#  TP=2                 : 2xH200 = 2x KV cache. KV is the concurrency limit here
#                         (32 KiB/token for this hybrid-attention 9B), and one
#                         backend must feed several CPU client jobs at once.
#  prefix caching       : each problem is sent n_samples times with an identical
#                         prompt (5, and 32 for the Research top-up), and several
#                         clients hit the same problems. Free prefill dedupe.
#                         Safe to enable per-backend: it changes cache reuse, not
#                         sampled tokens.
#  penalty fast path    : presence_penalty=1.5 runs the penalty kernel on every
#                         decode step; the stock path rebuilds a [batch, max_len]
#                         tensor each time.
#  MAX_NUM_SEQS=256     : with 2xH200 of KV there is room; the old 128 was chosen
#                         for a single 80G card. This is the knob that lets one
#                         backend serve many clients instead of queueing them.
#  chunked prefill      : already vLLM's default here; keeps a long prefill from
#                         stalling everyone else's decode.
export TP="${TP:-2}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FS_VLLM_PENALTY_FASTPATH=1
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"

echo "[pool] serving TAG=$TAG on ${HOSTNAME_FQ}:${PORT}  model=$MODEL"
echo "[pool] TP=$TP prefix_cache=$ENABLE_PREFIX_CACHING max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"
HOST=0.0.0.0 PORT="$PORT" MODEL_PATH="$MODEL" SERVED_MODEL_NAME="$TAG" \
  bash scripts/start_vllm_server.sh &
VLLM_PID=$!

# Only advertise once the server actually answers -- publishing early would hand
# clients a backend that 404s while weights are still loading.
ready=0
for _ in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[pool] vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "[pool] vLLM never became ready" >&2; exit 1; }

# Atomic publish: write to a temp file in the SAME directory, then rename. A
# reader can never observe a half-written entry.
tmp="$(mktemp "$REGISTRY/.tmp.XXXXXX")"
cat > "$tmp" <<JSON
{"tag": "$TAG", "host": "$HOSTNAME_FQ", "port": $PORT,
 "url": "http://${HOSTNAME_FQ}:${PORT}/v1", "model": "$MODEL",
 "job_id": "${SLURM_JOB_ID:-manual}"}
JSON
mv -f "$tmp" "$ENTRY"
echo "[pool] registered -> $ENTRY"

# Heartbeat: touch the entry so clients can age out a backend whose job was
# killed without running the trap (e.g. SIGKILL at the wall clock).
while kill -0 "$VLLM_PID" 2>/dev/null; do
  touch "$ENTRY" 2>/dev/null || true
  sleep 60
done
echo "[pool] vLLM exited; deregistering"

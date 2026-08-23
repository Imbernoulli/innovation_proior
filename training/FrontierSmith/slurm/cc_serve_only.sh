#!/bin/bash
#SBATCH --job-name=cc-serve-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gres=gpu:h100:2
#SBATCH --partition=pli
#SBATCH --account=goedelprover
#SBATCH --qos=pli-low
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
#
# SERVE-ONLY half of the split eval architecture (user ruling 2026-08-16):
# GPU jobs serve, and ONLY serve. Generation requests and all scoring live in
# separate CPU jobs (slurm/cc_eval_cpu_client.sh) -- scoring is CPU-bound by
# construction (FCS = go-judge compiling/running C++, Research = CPU subset,
# ALE = simulator), and the eval client itself is just HTTP + aggregation.
#
# This reuses the serve half of cc_eval_pool_selfhosted.sh verbatim (bind
# 0.0.0.0, probed port, atomic registry publish, 60s mtime heartbeat, dereg on
# exit) and adds the one thing a pure server must have: an answer to "when do I
# release the GPU?". Clients announce themselves by touching
#   $REGISTRY/<entry>.json.clients/<their job id>
# every 60s. This job checks every CHECK_INTERVAL_SEC (5 min) and exits when
#   - no client has EVER attached within SERVE_GRACE_SEC (30 min) of readiness, or
#   - clients did attach, but every heartbeat has been stale for
#     CLIENT_STALE_SEC (15 min) -- covers both clean exits (clients remove their
#     files) and killed clients (files stop being touched).
# Every check is logged, so a backend starving because the cpu partition won't
# schedule its clients is visible in the log, not a silent GPU burn.
#
# Preemption (measured 2026-08-19, job 12654964): pli-low serve jobs CAN be
# preempted mid-run (SIGTERM -> cleanup trap deregisters the entry cleanly).
# That is survivable by design: clients keep their samples.jsonl, so resubmit a
# serve job with the SAME TAG and rerun clients with RESUME=1 to backfill only
# the missing samples. Nothing scored is lost.
#
# Required env: MODEL (HF dir), TAG (served model name = registry tag).
# Optional: SERVE_GRACE_SEC, CLIENT_STALE_SEC, CHECK_INTERVAL_SEC, TP,
#           MAX_NUM_SEQS, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, ...
set -uo pipefail
PROJECT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$PROJECT_ROOT"
: "${MODEL:?set MODEL}" "${TAG:?set TAG}"

SERVE_GRACE_SEC="${SERVE_GRACE_SEC:-1800}"     # 30 min: max wait for a first client
CLIENT_STALE_SEC="${CLIENT_STALE_SEC:-900}"    # 15 min: heartbeat older than this = dead client
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}" # 5 min between client censuses

# ---- serve (identical to cc_eval_pool_selfhosted.sh's serve half) -------------
REGISTRY="${VLLM_POOL_REGISTRY:-$PROJECT_ROOT/.cache/vllm_pool}"
mkdir -p "$REGISTRY"
# Port selection must PROBE, not hash. Deriving it as job_id % N looks unique but
# collides whenever two concurrent jobs' ids are N apart -- job 12517854 died with
# "Address already in use" for exactly that reason. Ask the OS for a free port.
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
CLIENTS_DIR="${ENTRY}.clients"

# Throughput knobs, same as the pool/selfhosted servers.
export TP="${TP:-2}" ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FS_VLLM_PENALTY_FASTPATH=1
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"

cleanup() {
  rm -f "$ENTRY" >/dev/null 2>&1 || true
  rm -rf "$CLIENTS_DIR" >/dev/null 2>&1 || true
  [ -n "${TOUCH_PID:-}" ] && kill "$TOUCH_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}"  ] && kill "$VLLM_PID"  >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[serve] TAG=$TAG on ${HOSTN}:${VLLM_PORT} TP=$TP prefix=$ENABLE_PREFIX_CACHING seqs=$MAX_NUM_SEQS"
echo "[serve] idle policy: grace=${SERVE_GRACE_SEC}s stale=${CLIENT_STALE_SEC}s check=${CHECK_INTERVAL_SEC}s"
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
mkdir -p "$CLIENTS_DIR"
tmp="$(mktemp "$REGISTRY/.tmp.XXXXXX")"
printf '{"tag":"%s","host":"%s","port":%s,"url":"http://%s:%s/v1","model":"%s","job_id":"%s"}\n' \
  "$TAG" "$HOSTN" "$VLLM_PORT" "$HOSTN" "$VLLM_PORT" "$MODEL" "${SLURM_JOB_ID:-manual}" > "$tmp"
mv -f "$tmp" "$ENTRY"
echo "[serve] registered $ENTRY"
( while kill -0 "$VLLM_PID" 2>/dev/null; do touch "$ENTRY" 2>/dev/null || true; sleep 60; done ) &
TOUCH_PID=$!

# ---- idle watchdog: hold the GPU only while someone is (or may soon be) using it
ready_ts=$(date +%s)
last_active_ts=$ready_ts
ever_had_client=0
last_check=0
exit_reason=""
while :; do
  kill -0 "$VLLM_PID" 2>/dev/null || { exit_reason="vLLM process exited"; break; }
  now=$(date +%s)
  if (( now - last_check >= CHECK_INTERVAL_SEC )); then
    last_check=$now
    total=0; fresh=0
    for f in "$CLIENTS_DIR"/*; do
      [ -f "$f" ] || continue
      total=$((total+1))
      mt=$(stat -c %Y "$f" 2>/dev/null) || continue
      if (( now - mt <= CLIENT_STALE_SEC )); then fresh=$((fresh+1)); fi
    done
    if (( fresh > 0 )); then ever_had_client=1; last_active_ts=$now; fi
    up=$(( now - ready_ts ))
    echo "[serve] $(date '+%F %T') uptime=${up}s clients: total=$total fresh=$fresh"
    if (( fresh == 0 )); then
      if (( ever_had_client == 0 )); then
        if (( up >= SERVE_GRACE_SEC )); then
          exit_reason="no client ever attached within ${SERVE_GRACE_SEC}s grace period"
          break
        fi
        echo "[serve] STARVATION-WATCH: no client yet; holding GPU another $(( SERVE_GRACE_SEC - up ))s (are the cpu-partition clients still queued?)"
      else
        idle=$(( now - last_active_ts ))
        if (( idle >= CLIENT_STALE_SEC )); then
          exit_reason="all client heartbeats stale or removed for ${idle}s"
          break
        fi
        echo "[serve] all clients stale; ${idle}s idle of ${CLIENT_STALE_SEC}s allowed"
      fi
    fi
  fi
  sleep 30
done

echo "[serve] shutting down: ${exit_reason}"
exit 0

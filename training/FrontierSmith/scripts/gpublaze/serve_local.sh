#!/usr/bin/env bash
# Local (no-slurm) replacement for the GPU serve half of the split eval
# architecture (slurm/cc_serve_only.sh + scripts/vllm_pool_serve.sh).
#
#   GPUS=0,1 bash scripts/gpublaze/serve_local.sh <MODEL_PATH_or_HF_id> [TAG]
#
# Reuses scripts/start_vllm_server.sh UNCHANGED (it is env-driven and derives
# PROJECT_ROOT from its own location). What this wrapper replaces:
#   - sbatch/GRES            -> GPUS env (CUDA_VISIBLE_DEVICES), guarded so the
#                               other user's GPUs 6/7 are never touched
#   - GPFS registry           -> $FS_ROOT/.cache/vllm_pool on local disk
#   - <node>:<port> discovery -> 127.0.0.1:<port> (clients are on this box)
# The registry protocol itself (atomic publish, 60s mtime heartbeat, dereg on
# exit) is kept byte-compatible so scripts/vllm_pool_pick.py and the historical
# clients work as-is.
#
# Foreground by default; DAEMON=1 re-execs itself under setsid+nohup and returns.
# Optional env: TAG, PORT, TP (default = #GPUS), MAX_MODEL_LEN, MAX_NUM_SEQS,
#               GPU_MEMORY_UTILIZATION, VLLM_VENV, EXTRA_VLLM_ARGS,
#               IDLE_EXIT=1 (mirror cc_serve_only.sh's client-heartbeat watchdog:
#               exit when no client heartbeat for CLIENT_STALE_SEC after a grace)
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_gpublaze.sh"

MODEL="${1:?usage: GPUS=0,1 serve_local.sh <MODEL_PATH_or_HF_id> [TAG]}"
TAG="${2:-${TAG:-$(basename "$MODEL")}}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1 (GPUs 6,7 belong to another user -- refused)}"
fs_guard_gpus "$GPUS" || exit 1

if [ "${DAEMON:-0}" = "1" ]; then
  LOG="$FS_ROOT/logs/serve_${TAG}_$(date +%Y%m%d_%H%M%S).log"
  DAEMON=0 setsid nohup bash "${BASH_SOURCE[0]}" "$MODEL" "$TAG" >>"$LOG" 2>&1 &
  echo "[serve-local] daemonized (pid $!) log=$LOG"
  exit 0
fi

NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
export TP="${TP:-$NGPU}"
export CUDA_VISIBLE_DEVICES="$GPUS"

# Probe, don't hash (same rule as cc_serve_only.sh).
if [ -z "${PORT:-}" ]; then
  PORT=$("$FS_CLIENT_VENV/bin/python" -c 'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0",0)); print(s.getsockname()[1]); s.close()')
fi
HOSTN=127.0.0.1
ENTRY="$VLLM_POOL_REGISTRY/${TAG}__${HOSTN}__${PORT}.json"
CLIENTS_DIR="${ENTRY}.clients"
PIDFILE="$VLLM_POOL_REGISTRY/${TAG}__${HOSTN}__${PORT}.pid"

# Same throughput knobs as the pool servers, sized for H100-80G (the Princeton
# defaults assumed 2xH200):
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export FS_VLLM_PENALTY_FASTPATH="${FS_VLLM_PENALTY_FASTPATH:-1}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"

cleanup() {
  rm -f "$ENTRY" "$PIDFILE" >/dev/null 2>&1 || true
  rm -rf "$CLIENTS_DIR" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill -- -"$VLLM_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[serve-local] TAG=$TAG model=$MODEL gpus=$GPUS tp=$TP port=$PORT venv=$VLLM_VENV"
# shellcheck disable=SC2086
HOST=0.0.0.0 PORT="$PORT" MODEL_PATH="$MODEL" SERVED_MODEL_NAME="$TAG" \
  setsid bash "$FS_ROOT/scripts/start_vllm_server.sh" ${EXTRA_VLLM_ARGS:-} &
VLLM_PID=$!
echo "$VLLM_PID" > "$PIDFILE"

ready=0
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[serve-local] vLLM died during startup" >&2; exit 1; }
done
[ "$ready" = 1 ] || { echo "[serve-local] vLLM never became ready" >&2; exit 1; }

mkdir -p "$CLIENTS_DIR"
tmp="$(mktemp "$VLLM_POOL_REGISTRY/.tmp.XXXXXX")"
printf '{"tag":"%s","host":"%s","port":%s,"url":"http://%s:%s/v1","model":"%s","job_id":"local-%s"}\n' \
  "$TAG" "$HOSTN" "$PORT" "$HOSTN" "$PORT" "$MODEL" "$$" > "$tmp"
mv -f "$tmp" "$ENTRY"
echo "[serve-local] registered $ENTRY"

SERVE_GRACE_SEC="${SERVE_GRACE_SEC:-1800}"
CLIENT_STALE_SEC="${CLIENT_STALE_SEC:-900}"
ready_ts=$(date +%s); last_active_ts=$ready_ts; ever_had_client=0
while kill -0 "$VLLM_PID" 2>/dev/null; do
  touch "$ENTRY" 2>/dev/null || true
  if [ "${IDLE_EXIT:-0}" = "1" ]; then
    now=$(date +%s); fresh=0
    for f in "$CLIENTS_DIR"/*; do
      [ -f "$f" ] || continue
      mt=$(stat -c %Y "$f" 2>/dev/null) || continue
      (( now - mt <= CLIENT_STALE_SEC )) && fresh=$((fresh+1))
    done
    if (( fresh > 0 )); then ever_had_client=1; last_active_ts=$now; fi
    if (( fresh == 0 )); then
      if (( ever_had_client == 0 )) && (( now - ready_ts >= SERVE_GRACE_SEC )); then
        echo "[serve-local] no client within grace ${SERVE_GRACE_SEC}s; exiting"; break
      elif (( ever_had_client == 1 )) && (( now - last_active_ts >= CLIENT_STALE_SEC )); then
        echo "[serve-local] all client heartbeats stale ${CLIENT_STALE_SEC}s; exiting"; break
      fi
    fi
  fi
  sleep 30
done
echo "[serve-local] shutting down"

#!/usr/bin/env bash
# jiaolab (no-slurm) GPU serve half of the split eval architecture.
# Port of scripts/gpublaze/serve_local.sh; same registry protocol, same
# start_vllm_server.sh (UNCHANGED). Machine deltas:
#   - GPU police: gpublaze's static "GPU 6/7 belong to zzh" ban has no analogue
#     here; jiaolab's 8 A100s are shared with user `druv` on EVERY card, so the
#     guard is dynamic free-memory based (env_jiaolab.sh: fs_guard_gpus).
#   - A100-80G PCIe, NO NVLink: ONE TP=1 ENGINE PER CARD. TP=2 across PCIe on
#     the Qwen3.5 hybrid (attention + gated-delta-net) architecture is markedly
#     less efficient than two independent engines; the pool launcher
#     (launch_pool_eval.sh) is the supported way to use 2 cards.
#   - PREFIX CACHING OFF by default. Qwen3.5's mamba/GDN layers + vLLM 0.21
#     prefix caching wedge the engine (verified on gpublaze), so
#     ENABLE_PREFIX_CACHING defaults to 0 here and the pool launcher also passes
#     --no-enable-prefix-caching explicitly.
#   - VLLM_RPC_TIMEOUT=600000: 32k-token generations at CONCURRENCY=64 blow past
#     the stock RPC deadline.
#
#   GPUS=1 bash scripts/jiaolab/serve_local.sh <MODEL_PATH_or_HF_id> [TAG]
#
# Foreground by default; DAEMON=1 re-execs itself under setsid+nohup and returns.
# Optional env: TAG, PORT, TP (default = #GPUS), MAX_MODEL_LEN, MAX_NUM_SEQS,
#               GPU_MEMORY_UTILIZATION, VLLM_VENV, EXTRA_VLLM_ARGS, IDLE_EXIT=1.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_jiaolab.sh"

MODEL="${1:?usage: GPUS=1 serve_local.sh <MODEL_PATH_or_HF_id> [TAG]}"
TAG="${2:-${TAG:-$(basename "$MODEL")}}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=1 (cards are shared with user druv -- the guard refuses busy ones)}"
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
if [ "$TP" -gt 1 ]; then
  echo "[serve-local] WARNING: TP=$TP on A100 PCIe (no NVLink). One TP=1 engine per card is the verified jiaolab configuration; see launch_pool_eval.sh." >&2
fi

# Probe, don't hash (same rule as cc_serve_only.sh).
if [ -z "${PORT:-}" ]; then
  PORT=$("$FS_CLIENT_VENV/bin/python" -c 'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0",0)); print(s.getsockname()[1]); s.close()')
fi
HOSTN=127.0.0.1
ENTRY="$VLLM_POOL_REGISTRY/${TAG}__${HOSTN}__${PORT}.json"
CLIENTS_DIR="${ENTRY}.clients"
PIDFILE="$VLLM_POOL_REGISTRY/${TAG}__${HOSTN}__${PORT}.pid"

export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"   # Qwen3.5 GDN wedge; see header
export FS_VLLM_PENALTY_FASTPATH="${FS_VLLM_PENALTY_FASTPATH:-1}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
# 0.85 (gpublaze uses 0.90): jiaolab cards are SHARED with user druv (2-4G each,
# ~20G on GPU 0). 0.85 of 80G = 68G for our engine leaves ~9G of headroom so a
# co-tenant that grows mid-run is not OOM-killed by us. KV-cache size affects
# throughput only, never scores.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-41668}"

cleanup() {
  rm -f "$ENTRY" "$PIDFILE" >/dev/null 2>&1 || true
  rm -rf "$CLIENTS_DIR" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill -- -"$VLLM_PID" >/dev/null 2>&1 || true
  [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[serve-local] TAG=$TAG model=$MODEL gpus=$GPUS tp=$TP port=$PORT venv=$VLLM_VENV prefix_caching=$ENABLE_PREFIX_CACHING"
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

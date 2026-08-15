#!/bin/bash
# Supervisor for the Qwen3.8-27B TEACHER pass: re-roll every query Qwen3.6-27B never solved
# (unsolved.jsonl per domain: hard-fails that are not easy-drops and not solved by any pass).
# Schedule 4..32 (strong model — no need for 256), easy-threshold 1.1 => keep EVERY solve.
# temperature 1.0 per Qwen3.8 thinking-mode generation_config. Output: traces/<domain>.q38.jsonl
# (picked up by assemble_wave3.py). Points at the q38 service via --url/--model, NOT server.json.
set -u

SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/4fbbec36-23a3-4fd3-83db-61517e4405f7/scratchpad
REPO=/srv/home/bohanlyu/innovation_proior
VENV=/srv/home/bohanlyu/sesl/.venv
LOG="$SC/rollout_q38.log"
PORTS=${Q38_PORTS:-30002}
# comma-joined service URLs (one per TP=2 replica; driver pins queries across them)
URLS=$(for p in $PORTS; do printf "http://127.0.0.1:%s," "$p"; done); URLS=${URLS%,}
CONC=${Q38_CONC:-120}

STALL_SECS=${STALL_SECS:-21600}
CHECK_SECS=${CHECK_SECS:-60}

export TMPDIR="$SC"
export CUDA_VISIBLE_DEVICES=
mkdir -p "$SC"

trace_bytes() {
  find "$REPO/data_v4/_hardcp/traces" -maxdepth 1 -name '*.q38.jsonl' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1} END{print s+0}'
}

start_driver() {
  echo "[q38_watchdog $(date -u)] start python tools/hardcp_rollout.py --domains reasoning math code ifollow ioi ahc --worklist unsolved.jsonl --out-suffix .q38 --url $URLS --model Qwen3.8-27B --max-budget 64 --easy-threshold 1.1 --temperature 1.0 --max-tokens 57344 --request-timeout 3600 --concurrency $CONC --query-concurrency $((CONC+16)) --verify-workers 64" >> "$LOG"
  # shellcheck source=/srv/home/bohanlyu/sesl/.venv/bin/activate
  source "$VENV/bin/activate" || exit 1
  cd "$REPO" || exit 1
  python tools/hardcp_rollout.py \
    --domains reasoning math code ifollow ioi ahc \
    --worklist unsolved.jsonl \
    --out-suffix .q38 \
    --url "$URLS" \
    --model Qwen3.8-27B \
    --max-budget 64 \
    --easy-threshold 1.1 \
    --temperature 1.0 \
    --max-tokens 57344 \
    --request-timeout 3600 \
    --concurrency "$CONC" \
    --query-concurrency "$((CONC+16))" \
    --verify-workers 64 \
    >> "$LOG" 2>&1 &
  DRIVER_PID=$!
  LAST_BYTES=$(trace_bytes)
  LAST_CHANGE=$(date +%s)
  echo "[q38_watchdog $(date -u)] child pid=$DRIVER_PID trace_bytes=$LAST_BYTES" >> "$LOG"
}

stop_driver() {
  if [ "${DRIVER_PID:-0}" -le 1 ]; then
    return
  fi
  if kill -0 "$DRIVER_PID" 2>/dev/null; then
    kill "$DRIVER_PID" 2>/dev/null || true
    sleep 20
  fi
  if kill -0 "$DRIVER_PID" 2>/dev/null; then
    kill -9 "$DRIVER_PID" 2>/dev/null || true
  fi
  wait "$DRIVER_PID" 2>/dev/null || true
}

DRIVER_PID=0
LOCKDIR="$SC/driver_watchdog_q38.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[q38_watchdog $(date -u)] another q38_watchdog is already running; exiting" >> "$LOG"
  exit 0
fi
cleanup() {
  stop_driver
  rmdir "$LOCKDIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 0' INT TERM

LAST_BYTES=$(trace_bytes)
LAST_CHANGE=$(date +%s)
start_driver

while true; do
  sleep "$CHECK_SECS"
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    wait "$DRIVER_PID" 2>/dev/null
    rc=$?
    echo "[q38_watchdog $(date -u)] child exited rc=$rc -> restart" >> "$LOG"
    start_driver
    continue
  fi

  now=$(date +%s)
  bytes=$(trace_bytes)
  if [ "$bytes" != "$LAST_BYTES" ]; then
    LAST_BYTES="$bytes"
    LAST_CHANGE="$now"
    continue
  fi

  idle_for=$((now - LAST_CHANGE))
  if [ "$idle_for" -ge "$STALL_SECS" ]; then
    first_port=$(echo $PORTS | awk '{print $1}')
    up=$(curl -fs -o /dev/null -w "%{http_code}" --max-time 4 "http://127.0.0.1:$first_port/v1/models" 2>/dev/null)
    if [ "$up" = "200" ]; then
      echo "[q38_watchdog $(date -u)] no q38-trace growth for ${idle_for}s but service up -> restart child $DRIVER_PID" >> "$LOG"
      stop_driver
      start_driver
    fi
  fi
done

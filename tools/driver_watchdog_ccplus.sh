#!/bin/bash
# Supervisor for the CodeContests+ (ccplus) pass. ccplus sits at the TAIL of code/worklist.jsonl
# so the main first-pass won't reach it for a long time; this drives just the 793 ccplus problems
# (worklist_ccplus_only.jsonl) NOW, across all 3 vLLM services, writing to traces/code.ccplus.jsonl.
# Runs alongside driver A (code optim ahc first-pass) + driver B (deep reroll).
set -u

SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/4fbbec36-23a3-4fd3-83db-61517e4405f7/scratchpad
REPO=/srv/home/bohanlyu/innovation_proior
VENV=/srv/home/bohanlyu/sesl/.venv
LOG="$SC/rollout_ccplus.log"

STALL_SECS=${STALL_SECS:-2700}
CHECK_SECS=${CHECK_SECS:-60}

export TMPDIR="$SC"
export CUDA_VISIBLE_DEVICES=
mkdir -p "$SC"

trace_bytes() {
  find "$REPO/data_v4/_hardcp/traces" -maxdepth 1 -name 'code.ccplus.jsonl' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1} END{print s+0}'
}

running_reqs() {
  local total=0
  local port val
  for port in 30000 30001 30002; do
    val=$(curl -fs "http://127.0.0.1:$port/metrics" 2>/dev/null \
      | awk '!/^#/ && /num_requests_running/ {s+=$NF} END{print s+0}')
    total=$((total + ${val%.*}))
  done
  echo "$total"
}

start_driver() {
  echo "[ccplus_watchdog $(date -u)] start python tools/hardcp_rollout.py --domains code --worklist worklist_ccplus_only.jsonl --out-suffix .ccplus --max-budget 16 --concurrency 96 --query-concurrency 128 --verify-workers 48 (2 services, saturate)" >> "$LOG"
  # shellcheck source=/srv/home/bohanlyu/sesl/.venv/bin/activate
  source "$VENV/bin/activate" || exit 1
  cd "$REPO" || exit 1
  python tools/hardcp_rollout.py \
    --domains code \
    --worklist worklist_ccplus_only.jsonl \
    --out-suffix .ccplus \
    --max-budget 16 \
    --request-timeout 3600 \
    --concurrency 16 \
    --query-concurrency 32 \
    --verify-workers 48 \
    >> "$LOG" 2>&1 &
  DRIVER_PID=$!
  LAST_BYTES=$(trace_bytes)
  LAST_CHANGE=$(date +%s)
  echo "[ccplus_watchdog $(date -u)] child pid=$DRIVER_PID trace_bytes=$LAST_BYTES" >> "$LOG"
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
LOCKDIR="$SC/driver_watchdog_ccplus.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[ccplus_watchdog $(date -u)] another ccplus_watchdog is already running; exiting" >> "$LOG"
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
    echo "[ccplus_watchdog $(date -u)] child exited rc=$rc -> restart" >> "$LOG"
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
    running=$(running_reqs)
    if [ "$running" -eq 0 ]; then
      echo "[ccplus_watchdog $(date -u)] no ccplus-trace growth for ${idle_for}s and vLLM idle -> restart child $DRIVER_PID" >> "$LOG"
      stop_driver
      start_driver
    fi
  fi
done

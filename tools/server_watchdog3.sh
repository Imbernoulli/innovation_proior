#!/bin/bash
# Watchdog for the 2 independent TP=2 services (ports 30000/30001 on GPU pairs 1,2 / 3,5).
# If a port is down for two checks, kill its stale vllm (frees its GPUs) and relaunch just that one.
# Only touches OUR `vllm serve ... --port <p>` — never other users' sglang/GPUs.
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
REPO=/srv/home/bohanlyu/innovation_proior
LOCKDIR="$SC/server_watchdog3.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[watchdog $(date -u 2>/dev/null)] another server_watchdog3 is already running; exiting" >> "$SC/watchdog.log"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
declare -A GPUS=( [30000]="1,2" [30001]="3,5" )
up(){ curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
while true; do
  for port in 30000 30001; do
    if ! up "$port"; then
      sleep 10
      if ! up "$port"; then
        echo "[watchdog $(date -u 2>/dev/null)] port $port DOWN -> relaunch on GPUs ${GPUS[$port]}" >> "$SC/watchdog.log"
        pkill -9 -f "vllm serve Qwen/Qwen3.6-27B.*--port $port" 2>/dev/null; sleep 8
        bash "$REPO/tools/launch_1server.sh" "${GPUS[$port]}" "$port"
        sleep 210
      fi
    fi
  done
  sleep 30
done

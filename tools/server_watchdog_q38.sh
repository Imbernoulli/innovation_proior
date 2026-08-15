#!/bin/bash
# Server watchdog for the Qwen3.8 teacher services (port 30002 on GPUs 4,5; port 30003 on GPUs 6,7).
# If a port is down for two checks, kill its stale vllm and relaunch just that one.
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
REPO=/srv/home/bohanlyu/innovation_proior
LOCKDIR="$SC/server_watchdog_q38.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[q38-server-wd $(date -u 2>/dev/null)] another instance running; exiting" >> "$SC/watchdog_q38_server.log"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
declare -A GPUS=( [30000]="0,1" [30001]="2,3" [30002]="4,5" [30003]="6,7" )   # 4xTP=2, 40 in-flight each for Qwen3.8
up(){ curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
while true; do
  for port in 30000 30001 30002 30003; do
    if ! up "$port"; then
      sleep 10
      if ! up "$port"; then
        echo "[q38-server-wd $(date -u 2>/dev/null)] port $port DOWN -> relaunch on GPUs ${GPUS[$port]}" >> "$SC/watchdog_q38_server.log"
        # kill by port-specific numeric PIDs only (never pkill -f with a pattern in our own cmdline)
        for pid in $(ps -eo pid,args | awk -v P="--port $port" '$0 ~ /vllm serve Qwen\/Qwen3.8/ && index($0,P) {print $1}'); do
          kill -9 "$pid" 2>/dev/null
        done
        sleep 8
        bash "$REPO/tools/launch_1server_q38.sh" "${GPUS[$port]}" "$port"
        sleep 210
      fi
    fi
  done
  sleep 30
done

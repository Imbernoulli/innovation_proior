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
declare -A GPUS=( [30000]="0,1" [30001]="2,3" [30002]="4,5" )   # 3xTP=2 (GPUs 6,7 released to others), 30 in-flight each
up(){ curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
while true; do
  for port in 30000 30001 30002; do
    if ! up "$port"; then
      sleep 10
      if ! up "$port"; then
        echo "[q38-server-wd $(date -u 2>/dev/null)] port $port DOWN -> relaunch on GPUs ${GPUS[$port]}" >> "$SC/watchdog_q38_server.log"
        # Kill the FULL 3-layer tree (launcher -> EngineCore -> Worker_TP*). Killing only the launcher
        # orphans EngineCore+workers which keep ~75GB and the relaunch fails "Free memory ... 3/79 GiB".
        # Holders = every compute proc on this port's GPUs (+ their parents) + the port's launcher.
        gpus_csv="${GPUS[$port]}"
        {
          for g in ${gpus_csv//,/ }; do
            U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
            nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v u="$U" '$2==u{print $1}'
          done
          ps -eo pid,args | awk -v P="--port $port" '$0 ~ /vllm serve Qwen\/Qwen3.8/ && index($0,P) {print $1}'
        } | sort -u > "$SC/_kill_$port.txt"
        for pid in $(cat "$SC/_kill_$port.txt"); do ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' '; done | sort -u >> "$SC/_kill_$port.txt"
        for pid in $(sort -u "$SC/_kill_$port.txt"); do [ -n "$pid" ] && [ "$pid" != "1" ] && kill -9 "$pid" 2>/dev/null; done
        sleep 10
        # verify the GPUs are actually free before relaunching (else the relaunch just OOMs again)
        for g in ${gpus_csv//,/ }; do
          used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1==g{print $2}')
          [ "${used:-99999}" -gt 3000 ] && echo "[q38-server-wd $(date -u)] WARN GPU $g still ${used}MiB used after kill; relaunch may OOM" >> "$SC/watchdog_q38_server.log"
        done
        bash "$REPO/tools/launch_1server_q38.sh" "${GPUS[$port]}" "$port"
        sleep 210
      fi
    fi
  done
  sleep 30
done

#!/bin/bash
# Watcher for the Qwen3.8 teacher pass (live on port 30002, GPUs 4,5) + future GPU windows.
# Emits ONE LINE per event:
#   - q38 service down / recovered
#   - q38 driver dead
#   - q38 stall: no *.q38.jsonl growth > 45 min while service healthy
#   - preemption spike on 30002 (>100 per 3-min interval)
#   - free-GPU set changes (for grabbing MORE cards)
#   - hourly digest: per-domain q38 progress + in-flight + KV
REPO=/srv/home/bohanlyu/innovation_proior
TR=$REPO/data_v4/_hardcp/traces
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/4fbbec36-23a3-4fd3-83db-61517e4405f7/scratchpad

q38_bytes() { find "$TR" -maxdepth 1 \( -name '*.q38.jsonl' -o -name '*.q38b.jsonl' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
svc_up() { curl -fs -o /dev/null --max-time 4 "http://127.0.0.1:$1/v1/models" 2>/dev/null; }
preempt() { local t=0 v; for p in 30000 30001 30002; do v=$(curl -fs --max-time 4 http://127.0.0.1:$p/metrics 2>/dev/null | awk '!/^#/ && /num_preemptions_total/ {print int($NF)}'); t=$((t+${v:-0})); done; echo $t; }
running() { local t=0 v; for p in 30000 30001 30002; do v=$(curl -fs --max-time 4 http://127.0.0.1:$p/metrics 2>/dev/null | grep -E '^vllm:num_requests_running' | awk '{print int($2)}'); t=$((t+${v:-0})); done; echo $t; }
free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$2<3000{printf "%s ",$1}'; }
# cgshop RETIRED 2026-08-19 (thinking never terminates); cp2 COMPLETE 151/151 -> 2 drivers: cfr1@110 code@80
q38_driver_alive() { [ "$(ps -eo args | grep -cE 'hardcp_rollout\.py.*out-suffix \.q38')" -ge 2 ]; }

last_bytes=$(q38_bytes); last_change=$(date +%s); last_pre=$(preempt); last_pre=${last_pre:-0}
down_30000=0; down_30001=0; down_30002=0; stalled=0; tick=0; fw=0; fw_reported=0
echo "[q38-watch] armed: q38 bytes=$last_bytes driver=$(q38_driver_alive && echo up || echo DOWN)"
while true; do
  sleep 180
  tick=$((tick+1))
  for p in 30000 30001 30002; do
    dvar="down_$p"
    if ! svc_up $p; then
      [ "${!dvar}" = "0" ] && { echo "[q38-watch] ALERT: q38 service $p DOWN"; eval "$dvar=1"; }
    else
      [ "${!dvar}" = "1" ] && { echo "[q38-watch] q38 service $p RECOVERED"; eval "$dvar=0"; }
    fi
  done
  if ! q38_driver_alive; then echo "[q38-watch] ALERT: fewer than 2 q38 drivers alive (cfr1@110 code@80)"; sleep 600; fi
  b=$(q38_bytes); now=$(date +%s)
  if [ "$b" != "$last_bytes" ]; then last_bytes=$b; last_change=$now; [ "$stalled" = "1" ] && { echo "[q38-watch] q38 trace growth RESUMED"; stalled=0; }; fi
  idle=$((now - last_change))
  if [ "$idle" -ge 7200 ] && [ "$stalled" = "0" ]; then
    echo "[q38-watch] ALERT: no q38 trace growth for $((idle/60))min, in-flight=$(running), service_down=$down_30002"
    stalled=1
  fi
  # --- zombie-driver detector: alive but asyncio loop wedged (ep_poll, threads collapse to <=6, CPU<1%).
  #     Seen 2026-08-13 (base driver, 9 days) and 2026-08-16 (q38, 6h). Emit once per wedge episode.
  for pid in $(ps -eo pid,args | awk '$2=="python" && $0 ~ /hardcp_rollout\.py/ {print $1}'); do
    nl=$(ps -o nlwp= -p $pid 2>/dev/null | tr -d ' '); cpu=$(ps -o %cpu= -p $pid 2>/dev/null | tr -d ' ' | cut -d. -f1)
    age=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' '); [ "${age:-0}" -lt 1800 ] && continue   # <30min = still loading worklists
    # a driver holding open HTTP conns to the servers is WAITING on generations, not wedged (both real
    # zombies had 0 conns); few threads + low cpu is normal while awaiting long gens before any verify ran
    nconn=$(ss -tnp 2>/dev/null | grep "pid=$pid," | grep -cE ':3000[0-9] '); [ "${nconn:-0}" -gt 0 ] && { eval "zc_$pid=0"; continue; }
    tag=$(ps -o args= -p $pid | grep -oE 'out-suffix \S+' | awk '{print $2}')
    if [ "${nl:-99}" -le 6 ] && [ "${cpu:-99}" -lt 1 ]; then
      zc="zc_$pid"; eval "cnt=\${$zc:-0}"; cnt=$((cnt+1)); eval "$zc=$cnt"
      [ "$cnt" -eq 4 ] && echo "[q38-watch] ALERT: ZOMBIE driver pid=$pid ($tag): threads=$nl cpu=$cpu% for ~12min — kill -9 it, watchdog relaunches"
    else
      eval "zc_$pid=0"
    fi
  done
  pr=$(preempt); pr=${pr:-0}; d=$((pr - last_pre)); last_pre=$pr
  [ "$d" -gt 100 ] && echo "[q38-watch] ALERT: preemption spike +$d in 3min on 30002 (KV thrash)"
  # free-GPU windows: GPUs 6,7 belong to another user's flapping job (user said leave them) — only
  # report a window when >=2 GPUs have stayed free for 5 consecutive ticks (~15 min), once per episode.
  free=$(free_gpus); nf=$(echo $free | wc -w)
  if [ "$nf" -ge 2 ]; then fw=$((fw+1)); else fw=0; fw_reported=0; fi
  if [ "$fw" -ge 5 ] && [ "${fw_reported:-0}" = "0" ]; then echo "[q38-watch] LAUNCH WINDOW (stable 15min): $nf free GPUs: $free"; fw_reported=1; fi
  if [ $((tick % 20)) -eq 0 ]; then
    line="[q38-watch] digest:"
    for f in "$TR"/*.q38.jsonl "$TR"/*.q38b.jsonl; do
      [ -f "$f" ] && line="$line $(basename "$f" .jsonl)=$(wc -l < "$f")"
    done
    echo "$line in-flight=$(running)"
  fi
done

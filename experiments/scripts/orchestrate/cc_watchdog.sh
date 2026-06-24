#!/usr/bin/env bash
# Self-healing watchdog for the matrix orchestrator. Runs on the login node via
# setsid+nohup (same as the orchestrator). Its whole job is so the USER never has
# to notice a stall again:
#   * orchestrator process died (and matrix not complete)  -> restart it
#   * orchestrator alive but 0 jobs for ~35 min            -> kick (restart) it
#   * matrix complete (done >= total)                      -> log & exit cleanly
# Heartbeat every cycle to cc_watchdog.log so a human can glance at health.
#
#   launch:  setsid nohup bash cc_watchdog.sh >/dev/null 2>&1 < /dev/null &
#   stop:    pkill -9 -f 'cc_watchdog[.]sh'
#   watch:   tail -f cc_watchdog.log
set +u
cd /scratch/gpfs/CHIJ/bohan/fs
PAT="cc_orchestrator[.]py"
WLOG=cc_watchdog.log
STATE=cc_orchestrator_state.json
CYCLE=300                 # 5 min between checks
ZERO_KICK=7              # consecutive 0-job cycles (~35min) before kicking a live-but-stuck orch

# double-launch guard: flock singleton (bulletproof; pgrep -f self-matches the
# launcher command line, which contains this script's name).
me=$$
exec 9>/tmp/cc_watchdog.lock
if ! flock -n 9; then echo "[watchdog] already running (lock held); exiting" >> "$WLOG"; exit 0; fi

wlog() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$WLOG"; }

# total task count (one-time): each --plan task prints a 'deps=' line
TOTAL=$(python3 cc_orchestrator.py --plan 2>/dev/null | grep -c "deps=")
[ "$TOTAL" -gt 0 ] 2>/dev/null || TOTAL=178
wlog "watchdog START (pid=$me) total_tasks=$TOTAL cycle=${CYCLE}s"

zero=0
while true; do
  # done count from state json (status == "done")
  done=$(python3 - <<PY 2>/dev/null
import json,os
s=json.load(open("$STATE")) if os.path.exists("$STATE") else {}
print(sum(1 for v in s.values() if isinstance(v,dict) and v.get("status")=="done"))
PY
)
  [ -n "$done" ] || done=0
  njobs=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
  if pgrep -f "$PAT" >/dev/null; then alive=1; else alive=0; fi

  # completion: matrix done -> exit (do NOT restart a finished matrix)
  if [ "$done" -ge "$TOTAL" ] 2>/dev/null; then
    wlog "MATRIX COMPLETE done=$done/$TOTAL -> watchdog exiting"; break
  fi

  if [ "$alive" -eq 0 ]; then
    wlog "ORCHESTRATOR DOWN (done=$done/$TOTAL njobs=$njobs) -> restarting"
    bash cc_orch.sh start >> "$WLOG" 2>&1
    zero=0
  elif [ "$njobs" -eq 0 ]; then
    zero=$((zero+1))
    if [ "$zero" -ge "$ZERO_KICK" ]; then
      wlog "STUCK: orch alive but 0 jobs for ~$((zero*CYCLE/60))min -> kicking (restart)"
      bash cc_orch.sh restart >> "$WLOG" 2>&1
      zero=0
    else
      wlog "HEARTBEAT alive=1 njobs=0 done=$done/$TOTAL zero=$zero/$ZERO_KICK"
    fi
  else
    zero=0
    wlog "HEARTBEAT alive=1 njobs=$njobs done=$done/$TOTAL"
  fi
  sleep "$CYCLE"
done

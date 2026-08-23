#!/usr/bin/env bash
# Self-healing supervisor for the rlv10 campaign. Runs on a login node (nohup,
# setsid) and, every CHECK seconds, for each arm:
#   * arm running or queued  -> nothing
#   * arm gone AND its continuation chain gone -> RESUBMIT (one arm, chained x3)
#   * arm running but no new step for STALL_H hours -> dump diagnostics
#   * OOM / NaN grad in the last step -> loud line in the log
# Everything is appended to logs/rlv10_watchdog.log; a companion Monitor tails it.
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"
LOG="$FS/logs/rlv10_watchdog.log"
CHECK="${CHECK:-600}"
STALL_H="${STALL_H:-4}"
MAX_RESUBMIT="${MAX_RESUBMIT:-3}"      # per arm, over the watchdog's lifetime
ARMS=(base loraIM soupNEW10 soupWD03_20)
# Arms deliberately abandoned by a human (e.g. stuck in the truncation trap and
# superseded by a fixed-config rerun). The watchdog must NOT resurrect these:
# "gone with <TARGET_STEPS" is normally a failure, but not when it was a decision.
# Space-separated arm names; also honoured via the ABANDONED env var.
ABANDONED="${ABANDONED:-loraIM soupNEW10}"
declare -A RESUBMITS=()

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

resubmit() {  # arm
  local arm="$1"
  local n=${RESUBMITS[$arm]:-0}
  if [ "$n" -ge "$MAX_RESUBMIT" ]; then
    log "ALERT $arm: hit the resubmit cap ($MAX_RESUBMIT) -- NOT resubmitting, needs a human"
    return
  fi
  RESUBMITS[$arm]=$((n + 1))
  log "RESUBMIT $arm (attempt $((n + 1))/$MAX_RESUBMIT)"
  TAG=rlv10 ARMS_ONLY="$arm" bash "$FS/scripts/launch_rlv8.sh" >> "$LOG" 2>&1 || \
    log "ALERT $arm: resubmit command failed"
}

# Single-instance guard + heartbeat. The previous incarnation died silently and
# nobody noticed until an arm's whole continuation chain had failed with nothing
# resubmitting it. A heartbeat line every ~30 min makes "is it alive" checkable.
PIDF="$FS/logs/rlv10_watchdog.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "another watchdog is alive (pid $(cat "$PIDF")); exiting" >&2; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT
BEAT=0
log "watchdog start (pid $$): check=${CHECK}s stall=${STALL_H}h cap=${MAX_RESUBMIT}/arm"
while :; do
  for arm in "${ARMS[@]}"; do
    q=$(squeue -u "$USER" -h -o "%j %t" 2>/dev/null | grep -E "^rlv10-$arm(-c[23])? " || true)
    m="$FS/outputs/rl_multisource_rollout/rlv10_$arm/metrics.jsonl"

    if [ -z "$q" ]; then
      # nothing of this arm is queued or running. Finished on purpose?
      steps=$([ -f "$m" ] && wc -l < "$m" || echo 0)
      if [ "$steps" -ge "${TARGET_STEPS:-10}" ]; then
        log "DONE $arm: $steps steps reached, leaving it alone"
      elif echo " $ABANDONED " | grep -q " $arm "; then
        :   # deliberately stopped -- stay silent, do not resubmit
      else
        last=$(sacct -n -X --name="rlv10-$arm" -S now-2days -o State,Elapsed 2>/dev/null | tail -1 | tr -s ' ')
        log "GONE $arm after $steps steps (last: ${last:-unknown})"
        resubmit "$arm"
      fi
      continue
    fi

    # running: check for a stall
    if [ -f "$m" ]; then
      age_h=$(( ( $(date +%s) - $(stat -c %Y "$m") ) / 3600 ))
      if [ "$age_h" -ge "$STALL_H" ]; then
        node=$(squeue -u "$USER" -h -n "rlv10-$arm" -o "%N" | head -1)
        log "STALL $arm: no new step for ${age_h}h (node ${node:-?})"
        if [ -n "$node" ]; then
          {
            echo "  --- nvidia-smi ---"
            ssh -o ConnectTimeout=8 "$node" "nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader" 2>/dev/null
            echo "  --- host mem ---"
            ssh -o ConnectTimeout=8 "$node" "free -g | sed -n 2p" 2>/dev/null
          } >> "$LOG"
        fi
      fi
      # health of the most recent step
      tail -1 "$m" 2>/dev/null | python3 -c "
import json,sys
try:
    r=json.loads(sys.stdin.read())
except Exception: sys.exit()
g=r.get('actor/grad_norm')
if g is None or g!=g or g>5: print(f\"ALERT grad_norm={g} at step {r.get('step')}\")
" | while read -r line; do log "$arm: $line"; done
    fi
  done

  # eval-pipeline tripwire: ailab caps a user at 16 GPUs, which our 4 arms hold,
  # so evals queued there starve FOREVER without anything looking broken. Alert
  # if eval jobs exist but none has run for a long time.
  ev_tot=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "cc-eval" || true)
  ev_run=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -c "cc-eval" || true)
  if [ "${ev_tot:-0}" -gt 0 ] && [ "${ev_run:-0}" -eq 0 ]; then
    EV_STUCK=$((${EV_STUCK:-0} + 1))
    [ "$EV_STUCK" -ge 6 ] && { log "ALERT: $ev_tot eval job(s) queued, none running for ~1h -- check QOS/partition (ailab gres cap is 16/user)"; EV_STUCK=0; }
  else
    EV_STUCK=0
  fi

  # cluster-level tripwires
  oom=$(sacct -u "$USER" -X -S now-1hours -o JobName%22,State -n 2>/dev/null | grep -c "OUT_OF_ME" || true)
  [ "${oom:-0}" -gt 0 ] && log "ALERT: $oom job(s) hit OUT_OF_MEMORY in the last hour"

  BEAT=$((BEAT + 1))
  if [ $((BEAT % 3)) -eq 0 ]; then
    log "heartbeat: $(squeue -u "$USER" -h -o "%j" | grep -c "^rlv10-") rlv10 job(s) in queue, $(checkquota 2>/dev/null | grep -oE "[0-9.]+M +100M" | head -1 || echo "inodes ?")"
  fi
  sleep "$CHECK"
done

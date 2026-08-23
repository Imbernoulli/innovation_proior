#!/usr/bin/env bash
# Backfill the remaining matrix jobs as the QOS submit cap frees up.
# Retries every RETRY_S seconds until no DRYRUN combos remain. Idempotent.
# Signal-resilient: ignore HUP/PIPE so a parent-shell context reset (which
# delivered exit 144 to earlier detached runs) cannot kill the loop mid-flight.
trap '' HUP PIPE
set -uo pipefail
FS_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
RETRY_S="${RETRY_S:-300}"
MAX_LOOPS="${MAX_LOOPS:-60}"
LOG="$FS_ROOT/logs/theta_ttt_backfill_$(date +%Y%m%d_%H%M%S).log"
PAUSE_FLAG="$FS_ROOT/.theta_backfill_paused"
# Manual reprioritization lever: while this flag exists, do NOT submit (the theta
# matrix is lower user-priority than MLS/LoRA and was starving them via higher
# Slurm priority). Remove the flag + `scontrol release` the held jobs to resume.
[ -f "$PAUSE_FLAG" ] && { echo "backfill PAUSED at start ($PAUSE_FLAG); exiting" | tee -a "$LOG"; exit 0; }
echo "backfill loop start $(date)" | tee -a "$LOG"
for i in $(seq 1 "$MAX_LOOPS"); do
  [ -f "$PAUSE_FLAG" ] && { echo "[$(date +%H:%M:%S)] backfill PAUSED via flag; exiting" | tee -a "$LOG"; exit 0; }
  remaining=$(SKIP_EXISTING=1 DRYRUN=1 PORT_BASE=9000 PORT_STRIDE=20 bash "$FS_ROOT/slurm/cc_submit_theta_ttt_alltasks_matrix.sh" 2>/dev/null | grep -cE '^\[DRY\]')
  echo "[$(date +%H:%M:%S)] loop $i: remaining=$remaining" | tee -a "$LOG"
  if [ "$remaining" -eq 0 ]; then echo "ALL SUBMITTED" | tee -a "$LOG"; break; fi
  # randomized high base each retry (>= queued 8500-9380 range) + wide stride so
  # backfilled jobs never collide with queued jobs or each other on a shared node.
  RB=$(( 9000 + (RANDOM % 30) * 100 ))
  SKIP_EXISTING=1 PORT_BASE="$RB" PORT_STRIDE=20 bash "$FS_ROOT/slurm/cc_submit_theta_ttt_alltasks_matrix.sh" >>"$LOG" 2>&1 || true
  sleep "$RETRY_S"
done
echo "backfill loop end $(date)" | tee -a "$LOG"

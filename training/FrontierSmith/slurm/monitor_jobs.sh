#!/usr/bin/env bash
# Monitor a set of Slurm jobs and print status changes.
# Usage: bash slurm/monitor_jobs.sh JOBID1 JOBID2 ...

set -uo pipefail

LOG_FILE="${LOG_FILE:-logs/job_monitor.log}"
mkdir -p "$(dirname "$LOG_FILE")"

job_ids=("$@")
if [ ${#job_ids[@]} -eq 0 ]; then
  echo "Usage: $0 JOBID1 [JOBID2 ...]" >&2
  exit 1
fi

prev_states=""
while true; do
  states=$(squeue -j "$(IFS=,; echo "${job_ids[*]}")" -o '%.18i %.20j %.8T %.10M %R' -h 2>/dev/null)
  if [ "$states" != "$prev_states" ]; then
    echo "[$(date '+%F %T %Z')]" >> "$LOG_FILE"
    echo "$states" >> "$LOG_FILE"
    prev_states="$states"
  fi
  # Exit when no jobs remain in squeue (all done/cancelled)
  if [ -z "$states" ]; then
    echo "[$(date '+%F %T %Z')] All monitored jobs finished." >> "$LOG_FILE"
    sacct -j "$(IFS=,; echo "${job_ids[*]}")" --format=JobID,JobName,State,ExitCode,Elapsed >> "$LOG_FILE"
    exit 0
  fi
  sleep 120
done

#!/usr/bin/env bash
# Backfill the r1 theta/ttt matrix as the QOS submit cap frees. Idempotent (SKIP_EXISTING markers).
trap '' HUP PIPE
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
for i in $(seq 1 48); do
  # 45 = 5 models x 9 tasks; count submitted markers for r1_ models
  n=$(ls "$FS"/logs/.submitted_*_r1_* 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] backfill loop $i: r1 markers=$n/45"
  [ "$n" -ge 45 ] && { echo "ALL r1 theta/ttt submitted"; break; }
  SKIP_EXISTING=1 PORT_BASE=$(( 8200 + (RANDOM % 80) * 20 )) PORT_STRIDE=20 \
    bash "$FS/slurm/cc_submit_theta_ttt_r1_matrix.sh" >/dev/null 2>&1 || true
  sleep 300
done

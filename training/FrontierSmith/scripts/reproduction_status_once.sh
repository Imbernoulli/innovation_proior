#!/usr/bin/env bash
# Print one FrontierSmith reproduction status snapshot.
# This script intentionally does not loop or sleep.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

jobs=(
  9967723
  9977353
  9979101
  9983981
  9980304
  9980305
  9988286
  9988287
  9988288
  9980310
  9980311
  9980312
  9980313
  9980314
  9980315
  9980316
  9980317
  9980319
  9980320
  9980321
  9980322
  9980323
  9980324
  9980325
  9980326
  9980327
  9980328
  9980329
  9980330
  9992316
  9992317
  9993214
  9993215
  9993217
  9993218
  9993219
  9993220
  9993221
  9993222
  9993223
  9993224
  9993225
  9997403
  9997404
  9997748
  9997749
  9998167
  9998168
  9998169
  9998170
  9998171
  9998172
  10002286
  10002287
  10004498
  10004499
  10004506
  10004507
  10004508
  10007245
  10007246
  10009711
  10009712
)
job_csv="$(IFS=,; echo "${jobs[*]}")"

date '+%F %T %Z'
echo
echo "== squeue =="
squeue -j "$job_csv" -o '%.18i %.24j %.8T %.10M %.9l %.6D %R' || true
echo
echo "== sacct =="
sacct -j "$job_csv" --format=JobID,JobName%26,State,ExitCode,Elapsed,Start,End,NodeList%18 -P || true
echo
echo "== collected results =="
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/collect_reproduction_results.py
else
  python scripts/collect_reproduction_results.py
fi

#!/usr/bin/env bash
# Gate before committing hours to the full regeneration run.
#
# Samples a small set with the trained checkpoint and prints the QC table. Read it
# in this order — if 1 or 2 fails, the design is wrong and the full run is wasted:
#   1. 5-gram Jaccard  -> ~1.0 means the teacher is reciting the memorised think (no-op)
#   2. voice           -> scientist_voice must not collapse (baseline 31.0%),
#                         assistant_voice must not explode (baseline 11.4%)
#   3. hedges/dead-end -> should RISE (baseline 9.9% / 14.5%); this is wd01's defect
#   4. answer_terms_reached -> must not collapse (baseline 0.828), else the
#                         reasoning never actually reaches the answer
set -uo pipefail
cd /srv/home/bohanlyu/innovation_proior
N="${1:-24}"
C="${2:-24}"
OUT="${3:-.cache/tinker/gate_sample.jsonl}"
: "${TINKER_API_KEY:?set TINKER_API_KEY}"

rm -f "$OUT"
echo "=== sampling $N rows at concurrency $C with the trained checkpoint"
python3 experiments/scripts/tinker/sample_inkling.py \
    --state .cache/tinker/inkling_run.json \
    --limit "$N" --concurrency "$C" --max-tokens 16384 \
    --out "$OUT" 2>&1 | grep -vE "^Warning:|^\[transformers\]"

echo
echo "=== QC vs the hand-written original"
python3 experiments/scripts/tinker/qc_distill.py --distill "$OUT" --samples 3 \
    2>&1 | grep -vE "^Warning:|^\[transformers\]"

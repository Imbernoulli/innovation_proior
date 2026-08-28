#!/usr/bin/env bash
# One corrected-soup arm: GiantsBench 400 (the metric the soup lost on) +
# SciJudge no-thinking 1000 pairs (the clean judgement metric).
set -uo pipefail
cd /home/bohan/innovation_proior
TAG="$1"; PORT="$2"
P=training/FrontierSmith/.venv-jiaolab/bin/python
D=.cache/taste_eval
O=outputs_taste/run1
U="http://127.0.0.1:${PORT}/v1"
G="experiments/scripts/eval/taste/run_gen.py"

echo "=== [$TAG] giants $(date -Is)"
$P $G --bench giants --data $D/giants_test.parquet --n 400 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_giants.jsonl --max-tokens 32768 --concurrency 32

echo "=== [$TAG] scijudge_nothink $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_test.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge_nothink.jsonl --max-tokens 4096 --thinking off --concurrency 96

echo "=== [$TAG] ARM DONE $(date -Is)"

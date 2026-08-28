#!/usr/bin/env bash
# run_arm.sh <tag> <port>   -- runs the three taste benchmarks serially for one served model
set -uo pipefail
cd /home/bohan/innovation_proior
TAG="$1"; PORT="$2"
P=training/FrontierSmith/.venv-jiaolab/bin/python
D=.cache/taste_eval
O=outputs_taste/run1
mkdir -p "$O"
U="http://127.0.0.1:${PORT}/v1"
G="experiments/scripts/eval/taste/run_gen.py"
# NOTE: output filenames MUST start with cc_eval_<TAG>_ -- scripts/jiaolab/
# reap_finished_serves.sh looks for exactly that string on a client cmdline to
# decide the serve still has users; without it the engine is reaped after ~16min.

echo "=== [$TAG] soundness $(date -Is)"
$P $G --bench soundness --data $D/soundness.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_soundness.jsonl --max-tokens 16384 --concurrency 64

echo "=== [$TAG] giants $(date -Is)"
$P $G --bench giants --data $D/giants_test.parquet --n 400 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_giants.jsonl --max-tokens 32768 --concurrency 32

echo "=== [$TAG] scijudge $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_test.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge.jsonl --max-tokens 32768 --concurrency 48

echo "=== [$TAG] ALL DONE $(date -Is)"

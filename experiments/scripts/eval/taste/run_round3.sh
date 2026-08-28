#!/usr/bin/env bash
# Round 3 benches: PAIR-IQ (idea -> reviewer rating, swap-consistent, chance 25%)
# and SciPredict MCQ (natural-science experiment outcome, objective GT, post-2025-03).
# Both are judge-free and cheap, which is what makes them worth adding for power.
set -uo pipefail
ROOT="$1"; TAG="$2"; PORT="$3"; OUT="$4"; PY="$5"
cd "$ROOT"
D=.cache/taste_eval
U="http://127.0.0.1:${PORT}/v1"
G="experiments/scripts/eval/taste/run_gen.py"
mkdir -p "$OUT"
echo "=== [$TAG] pairiq $(date -Is)"
$PY $G --bench pairiq --data $D/pairiq --n 1000 --base-url "$U" --model "$TAG" \
     --out $OUT/cc_eval_${TAG}_pairiq.jsonl --max-tokens 16384 --concurrency 64 --profile thinking
echo "=== [$TAG] scipredict $(date -Is)"
$PY $G --bench scipredict --data $D/scipredict.csv --base-url "$U" --model "$TAG" \
     --out $OUT/cc_eval_${TAG}_scipredict.jsonl --max-tokens 16384 --concurrency 32 --profile thinking
echo "=== [$TAG] lit2test $(date -Is)"
$PY $G --bench lit2test --data $D/lit2test-benchmark --base-url "$U" --model "$TAG" \
     --out $OUT/cc_eval_${TAG}_lit2test.jsonl --max-tokens 16384 --concurrency 32 --profile thinking

echo "=== [$TAG] prescience $(date -Is)"
$PY $G --bench prescience --data $D/prescience_test.parquet --n 300 --base-url "$U" --model "$TAG" \
     --out $OUT/cc_eval_${TAG}_prescience.jsonl --max-tokens 16384 --concurrency 32 --profile thinking

echo "=== [$TAG] ROUND3 DONE $(date -Is)"

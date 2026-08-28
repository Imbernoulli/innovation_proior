#!/usr/bin/env bash
# Round 5: PRESCIENCE contribution generation (survey §29) + the two generation
# benches the 9B family has not run yet (HypoArena / AbGen).  All thinking=on.
set -uo pipefail
ROOT="$1"; TAG="$2"; PORT="$3"; OUT="$4"; PY="$5"
cd "$ROOT"; D=.cache/taste_eval; U="http://127.0.0.1:${PORT}/v1"
G="experiments/scripts/eval/taste/run_gen.py"; mkdir -p "$OUT"
run () { echo "=== [$TAG] $1 $(date -Is)"; shift; $PY $G "$@" --base-url "$U" --model "$TAG" --profile thinking; }
run prescience --bench prescience --data $D/prescience_test.parquet --n 300 \
    --out $OUT/cc_eval_${TAG}_prescience.jsonl --max-tokens 16384 --concurrency 32
run hypoarena  --bench hypoarena  --data $D/hypoarena.json --n 150 \
    --out $OUT/cc_eval_${TAG}_hypoarena.jsonl --max-tokens 16384 --concurrency 24
run abgen      --bench abgen      --data $D/abgen_test.json --n 200 \
    --out $OUT/cc_eval_${TAG}_abgen.jsonl --max-tokens 16384 --concurrency 32
echo "=== [$TAG] ROUND5 DONE $(date -Is)"

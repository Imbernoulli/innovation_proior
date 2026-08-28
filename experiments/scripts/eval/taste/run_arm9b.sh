#!/usr/bin/env bash
# One 9B arm, official Qwen3.5 sampling (presence_penalty 1.5).
# Output names keep the cc_eval_<TAG>_ prefix so reap_finished_serves.sh sees the client.
set -uo pipefail
cd /home/bohan/innovation_proior
TAG="$1"; PORT="$2"
P=training/FrontierSmith/.venv-jiaolab/bin/python
D=.cache/taste_eval
O=outputs_taste/run9b
mkdir -p "$O"
U="http://127.0.0.1:${PORT}/v1"
G="experiments/scripts/eval/taste/run_gen.py"

echo "=== [$TAG] rino $(date -Is)"
$P $G --bench rino --data $D/rino_test.parquet --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_rino.jsonl --max-tokens 16384 --concurrency 32 --profile thinking

echo "=== [$TAG] soundness $(date -Is)"
$P $G --bench soundness --data $D/soundness.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_soundness.jsonl --max-tokens 16384 --concurrency 64 --profile thinking

echo "=== [$TAG] giants $(date -Is)"
$P $G --bench giants --data $D/giants_test.parquet --n 400 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_giants.jsonl --max-tokens 32768 --concurrency 32 --profile thinking

echo "=== [$TAG] scijudge_nothink $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_test.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge_nothink.jsonl --max-tokens 4096 --thinking off \
     --concurrency 96 --profile instruct

echo "=== [$TAG] ARM DONE $(date -Is)"

#!/usr/bin/env bash
# run_arm2.sh <tag> <port>  -- round-2 taste benchmarks, cheap ones first.
# Waits for this tag's round-1 client to exit so the two rounds never overlap.
# Output names MUST keep the cc_eval_<TAG>_ prefix (reap_finished_serves.sh).
set -uo pipefail
cd /home/bohan/innovation_proior
TAG="$1"; PORT="$2"
P=training/FrontierSmith/.venv-jiaolab/bin/python
D=.cache/taste_eval
O=outputs_taste/run1
U="http://127.0.0.1:${PORT}/v1"
G="experiments/scripts/eval/taste/run_gen.py"

echo "=== [$TAG] round2 waiting for round1 to finish $(date -Is)"
while pgrep -f "run_gen.py .*--model $TAG .*_scijudge.jsonl" >/dev/null 2>&1; do sleep 60; done
while pgrep -f "run_arm.sh $TAG " >/dev/null 2>&1; do sleep 60; done

# No-thinking pass on the SAME 1,000 main-test pairs.  With thinking on these
# arms spend 17-18k tokens on a binary A/B call and 34-47% never terminate inside
# the 32k budget, which turns the score into a termination test.  The published
# anchor (Qwen3-4B-Instruct 58.1) is a non-thinking model, so this is also the
# directly comparable read.  Cheap: a few hundred tokens per item.
echo "=== [$TAG] scijudge_nothink $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_test.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge_nothink.jsonl --max-tokens 4096 --thinking off --concurrency 96

echo "=== [$TAG] rino $(date -Is)"
$P $G --bench rino --data $D/rino_test.parquet --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_rino.jsonl --max-tokens 16384 --concurrency 32

echo "=== [$TAG] abgen $(date -Is)"
$P $G --bench abgen --data $D/abgen_test.json --n 200 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_abgen.jsonl --max-tokens 16384 --concurrency 32

echo "=== [$TAG] hypoarena $(date -Is)"
$P $G --bench hypoarena --data $D/hypoarena.json --n 150 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_hypoarena.jsonl --max-tokens 16384 --concurrency 24

# Sub-sampled: at 32k budget these arms spend 7-12k tokens per binary judgement,
# so the full 904/611-pair splits would cost another ~10 GPU-hours per arm for a
# CI that is already tight enough at these sizes (+/- ~4pt).
echo "=== [$TAG] scijudge_iclr $(date -Is)"
$P $G --bench scijudge_iclr --data $D/scijudge_ood_iclr.jsonl --n 300 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge_iclr.jsonl --max-tokens 32768 --concurrency 64

echo "=== [$TAG] scijudge_ood_year $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_ood_year.jsonl --n 500 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge_oodyear.jsonl --max-tokens 32768 --concurrency 64

echo "=== [$TAG] ROUND2 DONE $(date -Is)"

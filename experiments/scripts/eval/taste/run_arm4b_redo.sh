#!/usr/bin/env bash
# 4B suite re-run under the OFFICIAL Qwen3.5 sampling profile (presence_penalty=1.5).
# The first pass used T=0.6 / presence_penalty=0 and spent 16-51% of samples in
# 25-gram loops; this re-run is the corrected-protocol version of that table.
# All three arms on gpublaze so the family stays internally comparable.
set -uo pipefail
cd /srv/home/bohanlyu/innovation_proior
TAG="$1"; MODEL="$2"; GPU="$3"
FS=training/FrontierSmith
P=$FS/.venv-gpublaze/bin/python
D=.cache/taste_eval
O=outputs_taste/run4b_redo
mkdir -p "$O"
G="experiments/scripts/eval/taste/run_gen.py"

echo "=== serve $TAG on GPU $GPU $(date -Is)"
( cd $FS && DAEMON=1 GPUS=$GPU MAX_MODEL_LEN=41668 bash scripts/gpublaze/serve_local.sh "$MODEL" "$TAG" )
for i in $(seq 1 240); do ls $FS/.cache/vllm_pool/${TAG}__*.json >/dev/null 2>&1 && break; sleep 10; done
PORT=$(ls $FS/.cache/vllm_pool/${TAG}__*.json 2>/dev/null | head -1 | sed 's/.*__//; s/\.json//')
[ -z "$PORT" ] && { echo "serve failed"; exit 1; }
U="http://127.0.0.1:${PORT}/v1"
echo "=== $TAG port $PORT"

echo "=== [$TAG] giants $(date -Is)"
$P $G --bench giants --data $D/giants_test.parquet --n 400 --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_giants.jsonl --max-tokens 32768 --concurrency 32 --profile thinking
echo "=== [$TAG] scijudge (thinking) $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_test.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge.jsonl --max-tokens 32768 --concurrency 64 --profile thinking
echo "=== [$TAG] rino $(date -Is)"
$P $G --bench rino --data $D/rino_test.parquet --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_rino.jsonl --max-tokens 16384 --concurrency 32 --profile thinking
echo "=== [$TAG] soundness $(date -Is)"
$P $G --bench soundness --data $D/soundness.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_soundness.jsonl --max-tokens 16384 --concurrency 64 --profile thinking
echo "=== [$TAG] scijudge_nothink $(date -Is)"
$P $G --bench scijudge --data $D/scijudge_test.jsonl --base-url "$U" --model "$TAG" \
     --out $O/cc_eval_${TAG}_scijudge_nothink.jsonl --max-tokens 4096 --thinking off \
     --concurrency 96 --profile instruct
( cd $FS && bash scripts/gpublaze/serve_stop.sh "$TAG" )
echo "=== [$TAG] REDO DONE $(date -Is)"

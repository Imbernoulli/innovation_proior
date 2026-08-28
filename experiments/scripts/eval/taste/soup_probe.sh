#!/usr/bin/env bash
# alpha-curve + rounding-noise control probe, all on gpublaze GPU 1, one engine at a time.
set -uo pipefail
cd /srv/home/bohanlyu/innovation_proior
FS=training/FrontierSmith
P=$FS/.venv-gpublaze/bin/python
BASE=$(ls -d /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*)
R=/srv/home/bohanlyu/models_sft/v2_multisetting_4b
OUT=outputs_taste/soup_probe
mkdir -p $OUT

run () {  # name path
  local NAME=$1 PATHM=$2
  echo "=== serve $NAME $(date -Is)"
  ( cd $FS && DAEMON=1 GPUS=1 MAX_MODEL_LEN=41668 bash scripts/gpublaze/serve_local.sh "$PATHM" "$NAME" )
  for i in $(seq 1 180); do
    ls $FS/.cache/vllm_pool/${NAME}__*.json >/dev/null 2>&1 && break
    sleep 10
  done
  local PORT=$(ls $FS/.cache/vllm_pool/${NAME}__*.json | head -1 | sed 's/.*__//; s/\.json//')
  echo "=== gen $NAME port=$PORT $(date -Is)"
  $P experiments/scripts/eval/taste/run_gen.py --bench giants \
     --data .cache/taste_eval/giants_test.parquet --n 400 --limit 150 \
     --base-url http://127.0.0.1:$PORT/v1 --model "$NAME" \
     --out $OUT/${NAME}.jsonl --max-tokens 32768 --concurrency 32
  ( cd $FS && bash scripts/gpublaze/serve_stop.sh "$NAME" )
  sleep 20
}

run probe_base      "$BASE"
run probe_soup_a10  $R/soup_full_wd01_a10
run probe_ulpctrl   $R/ulpctrl_a10
run probe_soup_a20  $R/soup_full_wd01_a20
run probe_wd01      $R/full_wd01
echo "=== PROBE DONE $(date -Is)"

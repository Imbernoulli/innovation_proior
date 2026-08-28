#!/usr/bin/env bash
# Serve a model, run round 5, stop.  <node> <tag> <model-path> <gpu>
set -uo pipefail
NODE="$1"; TAG="$2"; MODEL="$3"; GPU="$4"
if [ "$NODE" = jiaolab ]; then
  ROOT=/home/bohan/innovation_proior; PY=$ROOT/training/FrontierSmith/.venv-jiaolab/bin/python
  OUT=$ROOT/outputs_taste/run9b; SERVE=scripts/jiaolab/serve_local.sh
else
  ROOT=/srv/home/bohanlyu/innovation_proior; PY=$ROOT/training/FrontierSmith/.venv-gpublaze/bin/python
  OUT=$ROOT/outputs_taste/run4b_redo; SERVE=scripts/gpublaze/serve_local.sh
fi
cd "$ROOT"; FS=training/FrontierSmith
( cd $FS && DAEMON=1 GPUS=$GPU MAX_MODEL_LEN=41668 bash $SERVE "$MODEL" "$TAG" )
for i in $(seq 1 240); do ls $FS/.cache/vllm_pool/${TAG}__*.json >/dev/null 2>&1 && break; sleep 10; done
P=$(ls $FS/.cache/vllm_pool/${TAG}__*.json 2>/dev/null | head -1 | sed 's/.*__//; s/\.json//')
[ -z "$P" ] && { echo "serve failed"; exit 1; }
bash experiments/scripts/eval/taste/run_round5.sh "$ROOT" "$TAG" "$P" "$OUT" "$PY"
( cd $FS && bash $(dirname $SERVE)/serve_stop.sh "$TAG" ) 2>/dev/null || true

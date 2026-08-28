#!/usr/bin/env bash
# Run the round-3 benches after an arm's main suite finishes.
# Kept OUT of the arm scripts on purpose: those were already executing, and
# editing a running bash script shifts the byte offsets it reads from.
#   round3_followon.sh <node> <TAG> <marker-log> <models-dir-or-path>
set -uo pipefail
NODE="$1"; TAG="$2"; LOG="$3"; MODEL="${4:-}"
if [ "$NODE" = "jiaolab" ]; then
  ROOT=/home/bohan/innovation_proior; PY=$ROOT/training/FrontierSmith/.venv-jiaolab/bin/python
  OUT=$ROOT/outputs_taste/run9b; SERVE=scripts/jiaolab/serve_local.sh; DONE="ARM DONE"
else
  ROOT=/srv/home/bohanlyu/innovation_proior; PY=$ROOT/training/FrontierSmith/.venv-gpublaze/bin/python
  OUT=$ROOT/outputs_taste/run4b_redo; SERVE=scripts/gpublaze/serve_local.sh; DONE="REDO DONE"
fi
cd "$ROOT"; FS=training/FrontierSmith
echo "[r3:$TAG] waiting for '$DONE' in $LOG"
until grep -q "$DONE" "$LOG" 2>/dev/null; do sleep 60; done
P=$(ls $FS/.cache/vllm_pool/${TAG}__*.json 2>/dev/null | head -1 | sed 's/.*__//; s/\.json//')
if [ -z "$P" ]; then
  echo "[r3:$TAG] engine gone, re-serving"
  G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -F', ' '($3-$2)>=70000 {print $1; exit}')
  [ -z "$G" ] && { echo "[r3:$TAG] no free GPU"; exit 1; }
  ( cd $FS && DAEMON=1 GPUS=$G MAX_MODEL_LEN=41668 bash $SERVE "$MODEL" "$TAG" )
  for i in $(seq 1 240); do ls $FS/.cache/vllm_pool/${TAG}__*.json >/dev/null 2>&1 && break; sleep 10; done
  P=$(ls $FS/.cache/vllm_pool/${TAG}__*.json 2>/dev/null | head -1 | sed 's/.*__//; s/\.json//')
  [ -z "$P" ] && { echo "[r3:$TAG] re-serve failed"; exit 1; }
fi
bash experiments/scripts/eval/taste/run_round3.sh "$ROOT" "$TAG" "$P" "$OUT" "$PY"

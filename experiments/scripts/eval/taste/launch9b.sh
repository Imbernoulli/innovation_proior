#!/usr/bin/env bash
# Serve one 9B arm as soon as its weights land, then run the suite on it.
set -uo pipefail
cd /home/bohan/innovation_proior
TAG="$1"; MODEL="$2"; GPU="$3"
FS=training/FrontierSmith
echo "[$TAG] waiting for weights at $MODEL"
until [ -f "$MODEL/model.safetensors.index.json" ] || [ -f "$MODEL/model.safetensors" ]; do sleep 60; done
# let rsync settle: size must stop changing for two checks
prev=0
while :; do cur=$(du -sb "$MODEL" | cut -f1); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 45; done
echo "[$TAG] weights settled ($(du -sh "$MODEL" | cut -f1)); serving on GPU $GPU"
( cd $FS && DAEMON=1 GPUS=$GPU MAX_MODEL_LEN=41668 bash scripts/jiaolab/serve_local.sh "$MODEL" "$TAG" )
for i in $(seq 1 240); do ls $FS/.cache/vllm_pool/${TAG}__*.json >/dev/null 2>&1 && break; sleep 10; done
P=$(ls $FS/.cache/vllm_pool/${TAG}__*.json 2>/dev/null | head -1 | sed 's/.*__//; s/\.json//')
[ -z "$P" ] && { echo "[$TAG] serve failed"; exit 1; }
echo "[$TAG] port $P"
bash experiments/scripts/eval/taste/run_arm9b.sh "$TAG" "$P"

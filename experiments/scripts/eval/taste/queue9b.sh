#!/usr/bin/env bash
# Drain the 9B eval queue: for each queued model, wait for a free GPU, serve, run the suite.
set -uo pipefail
cd /home/bohan/innovation_proior
FS=training/FrontierSmith
Q=outputs_taste/queue9b.txt
DONE=outputs_taste/queue9b.done
MIN_FREE_MB=70000
touch "$Q" "$DONE"

free_gpu () {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
   | awk -F', ' -v m=$MIN_FREE_MB '($3-$2)>=m {print $1; exit}'
}

while :; do
  while read -r N; do
    [ -z "$N" ] && continue
    grep -qx "$N" "$DONE" && continue
    M=/home/bohan/models_hf/$N
    [ -f "$M/model.safetensors.index.json" ] || [ -f "$M/model.safetensors" ] || continue
    TAG="q9b_$(echo "$N" | sed 's/frontiersmith-q35-9b-//; s/[^A-Za-z0-9]/_/g')"
    G=""
    for i in $(seq 1 720); do G=$(free_gpu); [ -n "$G" ] && break; sleep 60; done
    [ -z "$G" ] && { echo "[queue] no free GPU for $TAG, retry later"; continue; }
    echo "[queue $(date +%H:%M:%S)] $TAG on GPU $G"
    setsid nohup bash experiments/scripts/eval/taste/launch9b.sh "$TAG" "$M" "$G" \
      > logs_taste/arm9b_${TAG}.log 2>&1 < /dev/null &
    echo "$N" >> "$DONE"
    sleep 300     # let the engine claim the card before scheduling the next one
  done < "$Q"
  sleep 120
done

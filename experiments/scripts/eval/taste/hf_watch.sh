#!/usr/bin/env bash
# Poll HF for new frontiersmith 9B checkpoints; download -> ship to jiaolab ->
# append to the jiaolab eval queue.  Runs on gpublaze (has the network + disk).
#   HF_TOKEN=... bash hf_watch.sh [poll_seconds]
set -uo pipefail
cd /srv/home/bohanlyu/innovation_proior
POLL="${1:-300}"
LOCAL=/srv/home/bohanlyu/models_hf
SEEN=$LOCAL/.seen
REMOTE=/home/bohan/models_hf
QUEUE=/home/bohan/innovation_proior/outputs_taste/queue9b.txt
V=training/FrontierSmith/.venv-gpublaze/bin/python
mkdir -p "$LOCAL"; touch "$SEEN"

while :; do
  REPOS=$(curl -s "https://huggingface.co/api/models?author=Bohan22&sort=lastModified&direction=-1&limit=40" \
          -H "Authorization: Bearer $HF_TOKEN" \
          | $V -c 'import sys,json;[print(m["id"]) for m in json.load(sys.stdin) if "frontiersmith" in m["id"]]' 2>/dev/null)
  for R in $REPOS; do
    N="${R##*/}"
    grep -qx "$N" "$SEEN" && continue
    echo "[watch $(date +%H:%M:%S)] new: $R"
    HF_HUB_OFFLINE=0 HF_HOME=/srv/home/bohanlyu/.cache/huggingface $V - <<PY || { echo "[watch] download failed $R"; continue; }
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id="$R", local_dir="$LOCAL/$N", token=os.environ["HF_TOKEN"],
                  allow_patterns=["*.json","*.txt","*.safetensors","*.jinja","tokenizer*","*.py","*.md"])
PY
    rsync -a "$LOCAL/$N/" "jiaolab:$REMOTE/$N/" || { echo "[watch] ship failed $N"; continue; }
    ssh jiaolab "mkdir -p \$(dirname $QUEUE); grep -qx '$N' $QUEUE 2>/dev/null || echo '$N' >> $QUEUE"
    echo "$N" >> "$SEEN"
    echo "[watch $(date +%H:%M:%S)] queued: $N"
  done
  sleep "$POLL"
done

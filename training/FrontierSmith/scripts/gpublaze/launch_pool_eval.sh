#!/usr/bin/env bash
# launch_pool_eval.sh <model_dir> <tag> <kind: fcsale|ale|fcs> <gpuA> <gpuB>
# Starts 2 TP=1 pool serves + 2 pinned clients (the arrangement that saturates
# both GPUs; least-loaded pick races when clients start together, so pin).
set -euo pipefail
M="$1"; TAG="$2"; KIND="$3"; GA="$4"; GB="$5"
SD="$(cd "$(dirname "$0")" && pwd)"; FS="$(cd "$SD/../.." && pwd)"
case "$KIND" in fcsale) SRC=both;; ale) SRC=alebench;; fcs) SRC=frontiercs;; *) echo bad KIND; exit 1;; esac
for g in "$GA" "$GB"; do
  FS_ALLOW_GPU67=1 GPUS=$g TP=1 TAG=$TAG DAEMON=1 IDLE_EXIT=1 VLLM_RPC_TIMEOUT=600000 \
    EXTRA_VLLM_ARGS="--no-enable-prefix-caching" bash "$SD/serve_local.sh" "$M" "$TAG" \
    > "$FS/logs/serve_pool_${TAG}_gpu$g.log" 2>&1
done
until [ "$(ls "$FS/.cache/vllm_pool/${TAG}"__*.json 2>/dev/null | wc -l)" -ge 2 ]; do sleep 15; done
P0=$(ls "$FS/.cache/vllm_pool/${TAG}"__*.json | sed -n 1p | grep -oE "[0-9]+\.json" | tr -d '.json')
P1=$(ls "$FS/.cache/vllm_pool/${TAG}"__*.json | sed -n 2p | grep -oE "[0-9]+\.json" | tr -d '.json')
for i in 0 1; do
  eval "P=\$P$i"
  TAG=$TAG MODEL_TAG=$TAG SOURCE=$SRC NUM_SHARDS=2 SHARD_IDX=$i REQUEST_TIMEOUT=7200 CONCURRENCY=64 \
    VLLM_BASE_URL=http://127.0.0.1:$P/v1 setsid nohup bash "$SD/eval_client_local.sh" \
    > "$FS/logs/cli_${TAG}_${SRC}${i}_pool.log" 2>&1 &
done
echo "[$TAG/$KIND] pool :$P0/:$P1 gpus $GA,$GB clients launched"

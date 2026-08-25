#!/usr/bin/env bash
# launch_pool_eval.sh <model_dir> <tag> <kind: fcsale|ale|fcs> [gpuA] [gpuB]
#
# The jiaolab production arrangement: 2 independent TP=1 pool serves + 2 clients
# PINNED one-per-engine. Port of scripts/gpublaze/launch_pool_eval.sh with the
# gpublaze-specific FS_ALLOW_GPU67 escape removed and the druv co-tenancy guard
# in its place.
#
# Why pinned and not --least-loaded: both clients start at the same instant, both
# ask the registry for the least-loaded backend, both get the SAME one, and one
# A100 sits idle for the whole run.
#
# Why TP=1 x2 and not TP=2: these are A100-80G PCIe cards with NO NVLink, and
# Qwen3.5 is a hybrid attention/gated-delta-net stack whose TP efficiency over
# PCIe is poor. Two independent engines beat one TP=2 engine here.
#
# GPUs: if gpuA/gpuB are omitted, the two freest cards are picked automatically.
# Every card is shared with user druv -- fs_guard_gpus refuses anything with less
# than FS_MIN_FREE_GB (default 70G) free, and we NEVER kill another user's job.
set -euo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
source "$SD/env_jiaolab.sh"
FS="$FS_ROOT"

M="${1:?usage: launch_pool_eval.sh <model_dir|hf_id> <tag> <fcsale|ale|fcs> [gpuA] [gpuB]}"
TAG="${2:?tag required}"
KIND="${3:-fcsale}"
if [ -n "${4:-}" ] && [ -n "${5:-}" ]; then
  GA="$4"; GB="$5"
else
  PICKED="$(fs_pick_free_gpus 2)" || { echo "no two free GPUs; try later or lower FS_MIN_FREE_GB" >&2; exit 1; }
  GA="${PICKED%,*}"; GB="${PICKED#*,}"
  echo "[$TAG] auto-picked GPUs $GA,$GB"
fi
fs_guard_gpus "$GA,$GB" || exit 1

case "$KIND" in
  fcsale) SRC=both;;
  ale)    SRC=alebench;;
  fcs)    SRC=frontiercs;;
  *) echo "bad KIND=$KIND (fcsale|ale|fcs)"; exit 1;;
esac

for g in "$GA" "$GB"; do
  GPUS=$g TP=1 TAG=$TAG DAEMON=1 IDLE_EXIT=1 VLLM_RPC_TIMEOUT=600000 \
    EXTRA_VLLM_ARGS="--no-enable-prefix-caching" bash "$SD/serve_local.sh" "$M" "$TAG" \
    > "$FS/logs/serve_pool_${TAG}_gpu$g.log" 2>&1
done

echo "[$TAG] waiting for 2 registered engines ..."
until [ "$(ls "$FS/.cache/vllm_pool/${TAG}"__*.json 2>/dev/null | wc -l)" -ge 2 ]; do sleep 15; done
P0=$(ls "$FS/.cache/vllm_pool/${TAG}"__*.json | sed -n 1p | grep -oE "[0-9]+\.json" | tr -d '.json')
P1=$(ls "$FS/.cache/vllm_pool/${TAG}"__*.json | sed -n 2p | grep -oE "[0-9]+\.json" | tr -d '.json')

for i in 0 1; do
  eval "P=\$P$i"
  TAG=$TAG MODEL_TAG=$TAG SOURCE=$SRC NUM_SHARDS=2 SHARD_IDX=$i \
    REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}" CONCURRENCY="${CONCURRENCY:-64}" \
    EVAL_SCORE_CONCURRENCY="${EVAL_SCORE_CONCURRENCY:-4}" \
    VLLM_BASE_URL=http://127.0.0.1:$P/v1 setsid nohup bash "$SD/eval_client_local.sh" \
    > "$FS/logs/cli_${TAG}_${SRC}${i}_pool.log" 2>&1 &
done
echo "[$TAG/$KIND] pool :$P0/:$P1 gpus $GA,$GB clients launched"
echo "  serve logs:  $FS/logs/serve_pool_${TAG}_gpu{$GA,$GB}.log"
echo "  client logs: $FS/logs/cli_${TAG}_${SRC}{0,1}_pool.log"
echo "  outputs:     $FS/outputs/cc_eval_${TAG}_*"

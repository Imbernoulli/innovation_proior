#!/usr/bin/env bash
# jiaolab one-command orchestrator (port of scripts/gpublaze/eval_split_local.sh)
# = 1 serve + K client processes on ONE card.
#
#   GPUS=1 bash scripts/jiaolab/eval_split_local.sh <MODEL_DIR_or_HF_id> [TAG] [KIND] [SHARDS]
#
#   KIND:   both (default) = FCS+ALE clients AND research clients
#           fcsale | research | ale
#   SHARDS: shards per kind (default 2)
#
# For the normal 2-card jiaolab arrangement (two independent TP=1 engines, one
# pinned client per engine) use scripts/jiaolab/launch_pool_eval.sh instead --
# it is what the anchor and all arm/soup runs use.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"

MODEL="${1:?usage: GPUS=1 eval_split_local.sh <MODEL> [TAG] [KIND] [SHARDS]}"
TAG="${2:-$(basename "$MODEL")}"
KIND="${3:-both}"
SHARDS="${4:-2}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=1}"
fs_guard_gpus "$GPUS" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- serve -------------------------------------------------------------------
if "$FS_CLIENT_VENV/bin/python" "$FS_ROOT/scripts/vllm_pool_pick.py" --tag "$TAG" >/dev/null 2>&1; then
  echo "[split-local] backend for TAG=$TAG already registered; reusing it"
  STARTED_SERVE=0
else
  GPUS="$GPUS" TAG="$TAG" IDLE_EXIT=1 DAEMON=1 VLLM_RPC_TIMEOUT=600000 \
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---no-enable-prefix-caching}" \
    bash "$SCRIPT_DIR/serve_local.sh" "$MODEL" "$TAG"
  STARTED_SERVE=1
fi

# ---- clients ------------------------------------------------------------------
plan=()
for shard in $(seq 0 $((SHARDS-1))); do
  case "$KIND" in
    both)     plan+=("both:$shard" "research:$shard") ;;
    fcsale)   plan+=("both:$shard") ;;
    research) plan+=("research:$shard") ;;
    ale)      plan+=("alebench:$shard") ;;
    *) echo "ERROR: unknown KIND=$KIND" >&2; exit 1 ;;
  esac
done

pids=(); logs=()
for spec in "${plan[@]}"; do
  src="${spec%%:*}"; shard="${spec##*:}"
  log="$FS_ROOT/logs/cli_${TAG}_${src}${shard}_${STAMP}.log"
  TAG="$TAG" MODEL_TAG="$TAG" SOURCE="$src" NUM_SHARDS="$SHARDS" SHARD_IDX="$shard" \
    bash "$SCRIPT_DIR/eval_client_local.sh" >"$log" 2>&1 &
  pids+=($!); logs+=("$log")
  echo "[split-local] client pid=$! src=$src shard=$shard/$SHARDS log=$log"
done

rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[split-local] CLIENT FAILED: ${logs[$i]}" >&2
    rc=1
  fi
done

# ---- teardown -----------------------------------------------------------------
if [ "${STARTED_SERVE}" = 1 ] && [ "${KEEP_SERVE:-0}" != "1" ]; then
  bash "$SCRIPT_DIR/serve_stop.sh" "$TAG" || true
fi
echo "[split-local] DONE rc=$rc  (outputs under $FS_ROOT/outputs/cc_eval_${TAG}_*)"
exit $rc

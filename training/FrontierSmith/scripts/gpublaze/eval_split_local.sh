#!/usr/bin/env bash
# Local one-command orchestrator = gpublaze replacement for
# slurm/cc_eval_split_submit.sh (1 GPU serve + K client processes).
#
#   GPUS=0,1 bash scripts/gpublaze/eval_split_local.sh <MODEL_DIR_or_HF_id> [TAG] [KIND] [SHARDS]
#
#   KIND:   both (default) = FCS+ALE clients AND research clients
#           fcsale | research | ale
#   SHARDS: shards per kind (default 2)
#
# Slurm dependency games (client_first vs serve_first) are pointless on one box:
# serve starts immediately, clients poll the registry. The serve runs with
# IDLE_EXIT=1 so it releases the GPUs the same way cc_serve_only.sh did once all
# client heartbeats go stale; this script also stops it explicitly at the end.
# All client logs land in $FS_ROOT/logs/.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_gpublaze.sh"

MODEL="${1:?usage: GPUS=0,1 eval_split_local.sh <MODEL> [TAG] [KIND] [SHARDS]}"
TAG="${2:-$(basename "$MODEL")}"
KIND="${3:-both}"
SHARDS="${4:-2}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
fs_guard_gpus "$GPUS" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- serve -------------------------------------------------------------------
if "$FS_CLIENT_VENV/bin/python" "$FS_ROOT/scripts/vllm_pool_pick.py" --tag "$TAG" >/dev/null 2>&1; then
  echo "[split-local] backend for TAG=$TAG already registered; reusing it"
  STARTED_SERVE=0
else
  GPUS="$GPUS" TAG="$TAG" IDLE_EXIT=1 DAEMON=1 bash "$SCRIPT_DIR/serve_local.sh" "$MODEL" "$TAG"
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

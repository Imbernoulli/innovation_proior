#!/usr/bin/env bash
# eval_queue.sh <queue-file>   *** RUN THIS ON jiaolab ***
#
# Why this exists: launch_pool_eval.sh needs two cards with >=70G free and
# refuses to touch druv's. With 6 serves already up there is usually exactly
# ONE free card, so firing evals by hand just prints "no two free GPUs" and
# drops the model on the floor. This drains a queue instead: it retries until
# a pair frees, launches, then backs off long enough for the new serves to
# claim those cards before it looks at the next model.
#
# Queue file: one "<model_dir_basename> <tag>" per line; # comments ok.
# Model dirs live under $JMODELS (shipped by eval_model_from_gpublaze.sh).
set -uo pipefail
FS="${FS_ROOT_OVERRIDE:-/home/bohan/innovation_proior/training/FrontierSmith}"
JMODELS="${JIAOLAB_MODELS:-/home/bohan/models_from_gpublaze}"
POLL="${POLL:-300}"          # how often to re-test for a free GPU pair
SETTLE="${SETTLE:-600}"      # let serves claim their cards before the next launch
cd "$FS"
Q="${1:?usage: eval_queue.sh <queue-file>}"
mkdir -p logs

while read -r MODEL TAG _rest; do
  [ -z "${MODEL:-}" ] && continue
  case "$MODEL" in \#*) continue ;; esac
  OUT="outputs/cc_eval_${TAG}_thinking_32k_both_vllm"
  if [ -d "$OUT" ]; then echo "[queue] $TAG: outputs exist, skip"; continue; fi
  if [ ! -d "$JMODELS/$MODEL" ]; then echo "[queue] $TAG: MISSING $JMODELS/$MODEL, skip"; continue; fi
  echo "[queue] $TAG: waiting for a free GPU pair ..."
  tries=0
  until bash scripts/jiaolab/launch_pool_eval.sh "$JMODELS/$MODEL" "$TAG" fcsale \
          >>"logs/queue_$TAG.log" 2>&1 </dev/null; do
    tries=$((tries+1))
    # A launch can fail for reasons no amount of waiting fixes (bad path, bad
    # tag). Only "no two free GPUs" is worth retrying; anything else, bail out
    # loudly rather than spinning forever on a broken entry.
    if ! tail -n 5 "logs/queue_$TAG.log" | grep -q "no two free GPUs"; then
      echo "[queue] $TAG: launch failed for a non-capacity reason; see logs/queue_$TAG.log"; break
    fi
    if [ "$tries" -ge "${MAX_TRIES:-288}" ]; then echo "[queue] $TAG: gave up after $tries tries"; break; fi
    sleep "$POLL"
  done
  if [ -d "$OUT" ] || grep -q "pool eval" "logs/queue_$TAG.log" 2>/dev/null; then
    echo "[queue] $TAG: launched; settling ${SETTLE}s"
    sleep "$SETTLE"
  fi
done < "$Q"
echo "[queue] drained $(date -Iseconds)"

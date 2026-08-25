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
# Queue file: one "<model_dir_basename> <tag> [kind]" per line; # comments ok.
# kind defaults to fcsale; use "ale" when only the ALE half needs redoing (the
# agentic-ablation a10 soups have clean 188-problem FrontierCS numbers already,
# but their ALE scored 24/40 and 1/40 during the dockerd outage).
# Model dirs live under $JMODELS (shipped by eval_model_from_gpublaze.sh).
set -uo pipefail
FS="${FS_ROOT_OVERRIDE:-/home/bohan/innovation_proior/training/FrontierSmith}"
JMODELS="${JIAOLAB_MODELS:-/home/bohan/models_from_gpublaze}"
POLL="${POLL:-300}"          # how often to re-test for a free GPU pair
SETTLE="${SETTLE:-600}"      # let serves claim their cards before the next launch
cd "$FS"
Q="${1:?usage: eval_queue.sh <queue-file>}"
mkdir -p logs

while read -r MODEL TAG KIND _rest; do
  [ -z "${MODEL:-}" ] && continue
  KIND="${KIND:-fcsale}"
  case "$MODEL" in \#*) continue ;; esac
  # Must match eval_client_local.sh's SOURCE->suffix table exactly; a wrong
  # suffix means the "outputs exist, skip" test never fires and the queue
  # relaunches the same model forever.
  case "$KIND" in
    ale)    SUF="_ale40_thinking_32k_vllm" ;;
    fcs)    SUF="_frontiercs_thinking_32k_vllm" ;;
    *)      SUF="_thinking_32k_both_vllm" ;;
  esac
  OUT="outputs/cc_eval_${TAG}${SUF}"
  if [ -d "$OUT" ]; then echo "[queue] $TAG: outputs exist, skip"; continue; fi
  if [ ! -d "$JMODELS/$MODEL" ]; then echo "[queue] $TAG: MISSING $JMODELS/$MODEL, skip"; continue; fi
  echo "[queue] $TAG: waiting for a free GPU pair ..."
  tries=0
  # Prefer two cards; fall back to a single card after FS_SINGLE_AFTER failed
  # rounds so a lone free GPU is used instead of sitting idle on a box we share
  # with another user. Single-engine runs are protocol-identical, just slower.
  try_launch() {
    bash scripts/jiaolab/launch_pool_eval.sh "$JMODELS/$MODEL" "$TAG" "$KIND" \
      >>"logs/queue_$TAG.log" 2>&1 </dev/null && return 0
    if [ "$tries" -ge "${FS_SINGLE_AFTER:-5}" ]; then
      g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F', ' '$1!=0 && $2<8000 {print $1; exit}')
      if [ -n "$g" ]; then
        echo "[queue] $TAG: only GPU $g free after $tries rounds -- single-engine fallback" >&2
        bash scripts/jiaolab/launch_pool_eval.sh "$JMODELS/$MODEL" "$TAG" "$KIND" "$g" "$g" \
          >>"logs/queue_$TAG.log" 2>&1 </dev/null && return 0
      fi
    fi
    return 1
  }
  until try_launch; do
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

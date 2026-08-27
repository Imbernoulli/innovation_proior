#!/usr/bin/env bash
# Upload the last two RL step-20 models to Bohan22, STRICTLY ONE AT A TIME.
#
# Serialization is not optional: creating a repo while an upload-large-folder is in
# flight tripped HF's "1000 api requests per 5 minutes" limit and killed the create.
# So each arm here is create -> upload -> wait -> next.
#
# Auth comes from ~/.cache/huggingface/token (0600). NEVER put the token on a command
# line: this is a shared login node and `ps` is world-readable.
#
# Model cards (README.md) are ALREADY written in each source dir. Do not regenerate.
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
export HF_HUB_DISABLE_TELEMETRY=1

do_one() {
  local repo="$1" src="$2" log="$3"
  echo "=== $(date -Is) $repo <- $src"
  if hf repo info "$repo" --repo-type model >/dev/null 2>&1; then
    echo "    repo exists, skipping create"
  else
    hf repo create "$repo" --repo-type model --private || { echo "    CREATE FAILED"; return 1; }
    sleep 10
  fi
  hf upload-large-folder "$repo" "$src" --repo-type=model --num-workers=8 >"$log" 2>&1
  local rc=$?
  echo "=== $(date -Is) $repo done rc=$rc"
  tail -3 "$log"
  return $rc
}

do_one Bohan22/frontiersmith-q35-9b-rl-soupWD03-20-step20 \
       "$D/models/rl_soupWD03_20_s20_hf" "$D/logs/hfup-rl-soupWD03.log"

echo "--- cooling down 120s before touching the API again (rate limit) ---"
sleep 120

do_one Bohan22/frontiersmith-q35-9b-rl-base-step20 \
       "$D/models/rl_base_s20_hf" "$D/logs/hfup-rl-base.log"

echo "=== ALL DONE $(date -Is)"

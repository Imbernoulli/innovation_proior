#!/usr/bin/env bash
# Submit the shard-aggregation job for a fast-chain eval, once -- and only once
# the shards are actually complete.
#
# Why this exists: eval_fast_chain.sh submits per-shard passes but NO aggregation
# step, because the pass count is not known in advance (a chain of N one-hour
# passes ends whenever the work runs out, not at a fixed job). The old submit
# path chained agg off the last generation job; the fast chain cannot.
#
# The failure this guards against is aggregating early. cc_eval_agg_shards.sh has
# its own EXPECTED_SAMPLES guard and refuses (it did, at 165/910), but a refused
# agg is a FAILED job in sacct that looks like a real failure -- so we only
# submit when the data is there.
#
# Completion test is the distinct-key count across shards >= EXPECTED_SAMPLES.
# The chain is submitted with more passes than it needs (the pass count is a
# guess made before we knew the throughput), so on completion we CANCEL the
# leftover passes rather than wait for them: each would otherwise spend ~5 min
# booting vLLM only to find zero work. Cancelling is safe once the count is
# reached -- any in-flight write can only produce a torn trailing line, which
# both this counter and _load_existing skip.
#
# Usage: TAGS="rlv10_base_s20 rlv10_base_s15" scripts/agg_when_complete.sh [fcsale|research]
#        (runs in the background; logs to logs/agg_when_complete.log)
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"
: "${TAGS:?set TAGS to a space-separated list of eval tags}"
KIND="${1:-fcsale}"
CHECK="${CHECK:-300}"
LOG="$FS/logs/agg_when_complete.log"

if [ "$KIND" = "research" ]; then
  SUFFIX=research_thinking_32k_vllm; EXPECTED=320; MAXERR=20; MODE=research
else
  SUFFIX=thinking_32k_both_vllm;     EXPECTED=910; MAXERR=12; MODE=fcsale
fi
# The defaults assume n_samples=5 (64x5=320, 182x5=910). A top-up run raises
# n_samples, so the target moves -- pass EXPECTED explicitly for those.
EXPECTED="${EXPECTED_OVERRIDE:-$EXPECTED}"
MAXERR="${MAXERR_OVERRIDE:-$MAXERR}"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# distinct-key count across both shards, matching the agg script's own dedupe
count_distinct() {  # output_base
  python3 - "$1" <<'PY'
import json, sys, pathlib
base = pathlib.Path(sys.argv[1])
keys = set()
for shard in ("shard_0", "shard_1"):
    p = base / shard / "samples.jsonl"
    if not p.exists():
        continue
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a pass killed mid-write leaves one torn line
            if r.get("error"):
                continue          # placeholder zeros are NOT completed work
            try:
                keys.add((r["data_source"], r["ground_truth"], int(r["sample_idx"])))
            except (KeyError, TypeError, ValueError):
                continue
print(len(keys))
PY
}

log "start: kind=$KIND tags='$TAGS' expected=$EXPECTED check=${CHECK}s"
remaining=($TAGS)
while [ "${#remaining[@]}" -gt 0 ]; do
  still=()
  for tag in "${remaining[@]}"; do
    ob="$FS/outputs/cc_eval_${tag}_${SUFFIX}"
    marker="$ob/.agg_submitted"
    if [ -e "$marker" ]; then log "$tag: already submitted ($(cat "$marker"))"; continue; fi

    # job IDs, not names: `scancel -n a,b,c` silently does nothing (learned the
    # hard way -- two arms ran 5h as zombies blocking the queue).
    live_ids=$(squeue -u "$USER" -h -o "%i|%j" 2>/dev/null | awk -F'|' -v t="$tag" '$2 ~ ("^cce-" t "-") {print $1}')
    live=$(echo "$live_ids" | grep -c . || true)
    n=$(count_distinct "$ob")
    n="${n:-0}"

    if [ "$n" -ge "$EXPECTED" ]; then
      if [ -n "$live_ids" ]; then
        log "$tag: complete at $n/$EXPECTED -> cancelling $live leftover pass(es): $(echo $live_ids)"
        scancel $live_ids 2>>"$LOG"
        sleep 5
      fi
      j=$(sbatch --parsable --job-name="cc-eval-9b-${tag}-agg" \
            --export=ALL,MODE=$MODE,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=$EXPECTED,MAX_ERRORS=$MAXERR \
            slurm/cc_eval_agg_shards.sh 2>>"$LOG")
      if [ -n "$j" ]; then
        echo "$j" > "$marker"
        log "$tag: COMPLETE ($n/$EXPECTED) -> agg job $j"
      else
        log "$tag: ALERT sbatch failed at $n/$EXPECTED"
        still+=("$tag")
      fi
    elif [ "${live:-0}" -eq 0 ] && [ "$n" -lt "$EXPECTED" ]; then
      # Chain exhausted its passes but the work is not done. A human must extend
      # it; silently waiting forever would hide this.
      log "$tag: ALERT chain finished but only $n/$EXPECTED samples -- needs more passes"
      still+=("$tag")
    else
      log "$tag: $n/$EXPECTED, $live pass(es) live"
      still+=("$tag")
    fi
  done
  remaining=("${still[@]}")
  [ "${#remaining[@]}" -eq 0 ] && break
  sleep "$CHECK"
done
log "all tags aggregated; exiting"

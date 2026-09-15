#!/usr/bin/env bash
# Submit the full coverage repair: every cell that has a stored generation but no score.
#   bash submit_repair.sh [frontiercs|alebench|frontiercs_research|all]
# Job shape deliberately matches the production eval client (-c 8, small worker count): FrontierCS
# TLE verdicts and ALE scores are wall-clock sensitive, so the repaired cells must be judged under
# the same CPU contention as the cells they sit next to. Parallelism comes from more JOBS, not from
# more workers inside a job.
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
KEYS="$D/rejudge/rejudge_keys.json"
WHICH="${1:-all}"
LOG="$D/logs/repair_jobids.txt"

groups() {  # split the arms that actually have work for $1 into $2 groups
  "$D/envs/client/bin/python" - "$1" "$2" <<'PY'
import json, sys
bench, n = sys.argv[1], int(sys.argv[2])
d = json.load(open("/scratch/gpfs/CHIJ/ziran/innov_v2_multi/rejudge/rejudge_keys.json"))
arms = sorted({k.split("|")[0]: len(v) for k, v in d.items() if k.endswith("|" + bench)}.items(),
              key=lambda kv: -kv[1])
buckets = [[] for _ in range(n)]; load = [0] * n
for arm, c in arms:                       # greedy balance by cell count
    i = load.index(min(load)); buckets[i].append(arm); load[i] += c
for b, l in zip(buckets, load):
    if b: print(" ".join(b), "#", l)
PY
}

submit() {  # bench ngroups walltime workers extra
  local bench=$1 n=$2 wall=$3 workers=$4 extra=${5:-}
  local i=0
  while read -r line; do
    [ -z "$line" ] && continue
    local arms="${line%%#*}" cells="${line##*# }"
    i=$((i+1))
    local j
    j=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G \
        --time="$wall" --job-name="rj-${bench:0:4}-$i" \
        --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.out" \
        --export=ALL,"BENCH=$bench,ARMS=$arms,KEYS=$KEYS,REJUDGE_WORKERS=$workers${extra:+,$extra}" \
        "$D/rejudge/rejudge_job2.sh") || { echo "sbatch failed: $bench $i"; continue; }
    echo "$(date +%F_%T) $bench g$i job=$j cells=$cells arms=$arms" | tee -a "$LOG"
  done < <(groups "$bench" "$n")
}

case "$WHICH" in
  frontiercs)          submit frontiercs          8 20:00:00 3 "REJUDGE_FCS_TIMEOUT=7200" ;;
  alebench)            submit alebench            2 08:00:00 3 ;;
  frontiercs_research) submit frontiercs_research 6 20:00:00 3 "REJUDGE_MODEL_ERR_AS_ZERO=1" ;;
  all) submit alebench 2 08:00:00 3
       submit frontiercs 8 20:00:00 3 "REJUDGE_FCS_TIMEOUT=7200"
       submit frontiercs_research 6 20:00:00 3 "REJUDGE_MODEL_ERR_AS_ZERO=1" ;;
  *) echo "usage: submit_repair.sh [frontiercs|alebench|frontiercs_research|all]"; exit 2;;
esac

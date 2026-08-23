#!/bin/bash
# One command = 1 GPU serve job + K CPU client jobs (the split eval architecture,
# user ruling 2026-08-16: GPU jobs only serve; generation requests + scoring run
# on the cpu partition).
#
#   bash slurm/cc_eval_split_submit.sh <MODEL_DIR> [TAG] [KIND] [SHARDS]
#
#   KIND: both (default) = FCS+ALE clients AND research clients
#         fcsale         = FCS+ALE only        research = research only
#         ale            = ALE full-40 only (set ALEBENCH_VAL_DATA yourself)
#   SHARDS: shards per kind (default 2)
#
# Ordering (DEP_MODE):
#   client_first (default) -- clients are submitted free; the serve job gets
#     --dependency="after:c1?after:c2..." so it only starts queueing for GPUs
#     once the FIRST client is already running and polling the registry
#     (POOL_WAIT covers the serve job's own queue+load time). Chosen default
#     because idle-waiting is then spent on cpu cores, never on GPUs: measured
#     2026-08-19, the cpu queue ran 1250 jobs deep while pli-low both queued
#     ~1.5h AND preempted a running serve job -- serve-first burned 1h45m of
#     2xH100 without a single client arriving.
#   serve_first -- serve submitted first, clients held with
#     --dependency=after:<serveJobId>. Use when the cpu partition is fast and
#     you want zero client-side polling. If the cpu queue backs up, raise the
#     serve grace: export SERVE_GRACE_SEC=7200 before submitting.
# Either way the serve job self-terminates: SERVE_GRACE_SEC (default 30 min)
# with no client ever attaching, or 15 min after every client heartbeat goes
# stale (see slurm/cc_serve_only.sh). A preempted/killed serve job is cheap to
# recover: clients keep their samples.jsonl, so resubmit a serve job with the
# same TAG and rerun clients with RESUME=1 (the default) to backfill.
#
# Timing fidelity (audit 2026-08-19/20): cpu-partition genoa nodes judge 15-20%
# slower than the ailab/pli hosts historical numbers came from -- near-threshold
# FCS solutions flip 100->0 on node class alone. Therefore:
#   - clients default to slurm/cc_eval_cpu_client_pinned.sh (gojudge_shim_v2,
#     SHIM_PIN_CORES: one exclusive core per judged run, contention-free timing;
#     CLIENT_VARIANT=plain falls back to cc_eval_cpu_client.sh);
#   - CLIENT_CONSTRAINT=<feature> (e.g. cascade, genoa) pins the node class --
#     set it whenever the numbers will sit in the same table as historical runs;
#   - every client writes judge_node_meta.json (node, features, cpu model, shim
#     nodeSpeedCalibration) next to each shard's samples.jsonl for later audit.
#
# Overridables (env): GPU_PARTITION/GPU_ACCOUNT/GPU_QOS/GPU_GRES/SERVE_TIME,
# CLIENT_CPUS/CLIENT_MEM/CLIENT_TIME/CLIENT_VARIANT/CLIENT_CONSTRAINT,
# SERVE_GRACE_SEC, POOL_WAIT, and every protocol/limit env the client
# understands (exported via --export=ALL).
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"

MODEL="${1:?usage: cc_eval_split_submit.sh <MODEL_DIR> [TAG] [KIND] [SHARDS]}"
TAG="${2:-$(basename "$MODEL")}"
KIND="${3:-both}"
SHARDS="${4:-2}"
[ -d "$MODEL" ] || { echo "ERROR: MODEL dir not found: $MODEL" >&2; exit 1; }

GPU_PARTITION="${GPU_PARTITION:-pli}"
GPU_ACCOUNT="${GPU_ACCOUNT:-goedelprover}"
GPU_QOS="${GPU_QOS:-pli-low}"
GPU_GRES="${GPU_GRES:-gpu:h100:2}"
SERVE_TIME="${SERVE_TIME:-12:00:00}"
SERVE_CPUS="${SERVE_CPUS:-16}"
SERVE_MEM="${SERVE_MEM:-220G}"
CLIENT_CPUS="${CLIENT_CPUS:-16}"
CLIENT_MEM="${CLIENT_MEM:-96G}"
CLIENT_TIME="${CLIENT_TIME:-08:00:00}"   # short jobs backfill the congested cpu queue far better
CLIENT_VARIANT="${CLIENT_VARIANT:-pinned}"   # pinned | plain
case "$CLIENT_VARIANT" in
  pinned) CLIENT_SCRIPT=slurm/cc_eval_cpu_client_pinned.sh ;;
  plain)  CLIENT_SCRIPT=slurm/cc_eval_cpu_client.sh ;;
  *) echo "ERROR: CLIENT_VARIANT must be pinned|plain" >&2; exit 1 ;;
esac
CLIENT_CONSTRAINT="${CLIENT_CONSTRAINT:-}"   # e.g. cascade / genoa; empty = any

DEP_MODE="${DEP_MODE:-client_first}"

submit_serve() {  # [dependency-spec]
  local dep=()
  [ -n "${1:-}" ] && dep=(--dependency="$1")
  sbatch --parsable --job-name="srv-${TAG}" "${dep[@]}" \
    --partition="$GPU_PARTITION" --account="$GPU_ACCOUNT" --qos="$GPU_QOS" \
    --gres="$GPU_GRES" -c "$SERVE_CPUS" --mem="$SERVE_MEM" --time="$SERVE_TIME" \
    --export=ALL,MODEL="$MODEL",TAG="$TAG" \
    slurm/cc_serve_only.sh || { echo "ERROR: serve sbatch failed" >&2; exit 1; }
}

submit_client() {  # source shortname shard [dependency-spec]
  local src="$1" short="$2" shard="$3" dep=()
  [ -n "${4:-}" ] && dep=(--dependency="$4")
  local jid
  local extra=()
  [ -n "$CLIENT_CONSTRAINT" ] && extra=(--constraint="$CLIENT_CONSTRAINT")
  jid=$(sbatch --parsable --job-name="cli-${TAG}-${short}${shard}" "${dep[@]}" "${extra[@]}" \
    --partition=cpu -c "$CLIENT_CPUS" --mem="$CLIENT_MEM" --time="$CLIENT_TIME" \
    --export=ALL,TAG="$TAG",MODEL_TAG="$TAG",SOURCE="$src",NUM_SHARDS="$SHARDS",SHARD_IDX="$shard" \
    "$CLIENT_SCRIPT") || { echo "ERROR: client sbatch failed" >&2; exit 1; }
  echo "client: $jid (cli-${TAG}-${short}${shard} src=$src shard=$shard/$SHARDS)"
  client_ids+=("$jid")
}

client_ids=()
plan=()
for shard in $(seq 0 $((SHARDS-1))); do
  case "$KIND" in
    both)     plan+=("both f $shard" "research r $shard") ;;
    fcsale)   plan+=("both f $shard") ;;
    research) plan+=("research r $shard") ;;
    ale)      plan+=("alebench a $shard") ;;
    *) echo "ERROR: unknown KIND=$KIND" >&2; exit 1 ;;
  esac
done

if [ "$DEP_MODE" = "serve_first" ]; then
  srv=$(submit_serve "")
  echo "serve: $srv (srv-${TAG}, $GPU_PARTITION $GPU_GRES)"
  for spec in "${plan[@]}"; do submit_client $spec "after:${srv}"; done
else
  for spec in "${plan[@]}"; do submit_client $spec ""; done
  # '?' = OR: serve becomes eligible as soon as ANY client is running.
  dep=""
  for j in "${client_ids[@]}"; do dep="${dep:+$dep?}after:$j"; done
  srv=$(submit_serve "$dep")
  echo "serve: $srv (srv-${TAG}, $GPU_PARTITION $GPU_GRES, starts once any client runs)"
fi
echo "done. serve self-terminates when clients finish (or after SERVE_GRACE_SEC with none)."

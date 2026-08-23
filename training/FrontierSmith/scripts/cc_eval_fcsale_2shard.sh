#!/usr/bin/env bash
# Submit a 2-way-sharded FCS/ALE eval (official avg@5 pipeline) for ANY 9B model dir.
# Same contract as cc_soup_sweep_eval.sh's submit_fcsale, but standalone: use it for
# models that are NOT soups (base, raw SFT, RL exports) or to backfill a single alpha.
#   bash cc_eval_fcsale_2shard.sh <tag> <model_dir>
# Routing defaults to ailab. NOTE the go-judge "no space left on device" failures that
# plagued FCS evals on 2026-07-28 were NOT partition-specific: the CHIJ GPFS fileset had
# hit its 80TiB quota, and writes returning EDQUOT surface as ENOSPC inside go-judge's
# container prefork. `df` does NOT show this (it reports the whole 14P filesystem, not the
# fileset quota) -- always check with `checkquota`. Partition is therefore not the fix;
# free space is. ailab is kept as the target because it is the script's native #SBATCH
# partition and the fastest (~2:15/shard vs ~3.7h on pli).
# Walls are 6h: a 172-problem x 5-sample shard takes ~3.7h on pli H100 (2:15 on ailab
# H200), so the old 3h wall TIMEOUTs at ~80%. RESUME=1 is the eval default, and the
# wave-2 job picks up any partial samples.jsonl.
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
TAG="${1:?need tag}"; MP="${2:?need model dir}"
[ -e "$MP/config.json" ] || { echo "ERROR: $MP has no config.json"; exit 1; }
EV_DEST="${EV_DEST:---partition=ailab --account=chij --gres=gpu:1}"
MEMFLAG="${MEMFLAG:---mem=110G}"
OB="$FS/outputs/cc_eval_${TAG}_thinking_32k_both_vllm"
COMMON="MODEL_PATH=$MP,TAG=$TAG,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
s0=$(sbatch --parsable $EV_DEST --time=06:00:00 $MEMFLAG --job-name="ev-${TAG}-s0" --export=ALL,$COMMON,SHARD_IDX=0,OUTPUT_DIR="$OB/shard_0",SAMPLES_JSONL="$OB/shard_0/samples.jsonl",SUMMARY_JSON="$OB/shard_0/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
s1=$(sbatch --parsable $EV_DEST --time=06:00:00 $MEMFLAG --job-name="ev-${TAG}-s1" --export=ALL,$COMMON,SHARD_IDX=1,OUTPUT_DIR="$OB/shard_1",SAMPLES_JSONL="$OB/shard_1/samples.jsonl",SUMMARY_JSON="$OB/shard_1/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
s0w2=$(sbatch --parsable $EV_DEST --time=03:00:00 $MEMFLAG --dependency=afterany:$s0 --job-name="ev-${TAG}-s0w2" --export=ALL,$COMMON,SHARD_IDX=0,OUTPUT_DIR="$OB/shard_0",SAMPLES_JSONL="$OB/shard_0/samples.jsonl",SUMMARY_JSON="$OB/shard_0/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
s1w2=$(sbatch --parsable $EV_DEST --time=03:00:00 $MEMFLAG --dependency=afterany:$s1 --job-name="ev-${TAG}-s1w2" --export=ALL,$COMMON,SHARD_IDX=1,OUTPUT_DIR="$OB/shard_1",SAMPLES_JSONL="$OB/shard_1/samples.jsonl",SUMMARY_JSON="$OB/shard_1/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="ev-${TAG}-agg" --export=ALL,MODE=fcsale,OUTPUT_BASE="$OB",EXPECTED_SAMPLES=910,MAX_ERRORS=12 slurm/cc_eval_agg_shards.sh >/dev/null
echo "fcsale $TAG -> s0=$s0 s1=$s1 (w2 $s0w2/$s1w2)"

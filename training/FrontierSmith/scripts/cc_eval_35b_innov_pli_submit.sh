#!/usr/bin/env bash
# Eval an already-HF 35B model (SFT recipe search) on pli H100 TP=2, OFF ailab so
# it never competes with the priority 35B SFT node. Mirrors the pli invocation in
# rl35b_ckpt_eval_waiter.sh (2-way sharded FCS/ALE + research, w2 resumes, CPU agg).
#   scripts/cc_eval_35b_innov_pli_submit.sh <tag> <model_dir> [fcsale|research|both]
set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
cd "$FS"

TAG="${1:?need tag}"; MP="${2:?need model dir}"; WHICH="${3:-both}"
[ -e "$MP/config.json" ] || { echo "ERROR: $MP has no config.json"; exit 1; }

EVAL_TP=2
PARTFLAGS=(--partition=pli --gres=gpu:2 --account=goedelprover --qos=pli-low)

submit_fcsale () {
  local ob="$FS/outputs/cc_eval_${TAG}_thinking_32k_both_vllm"
  local common="MODEL_PATH=$MP,TAG=$TAG,TP=$EVAL_TP,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --job-name="cc-eval-35b-${TAG}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --job-name="cc-eval-35b-${TAG}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s0w2=$(sbatch --parsable --time=03:30:00 "${PARTFLAGS[@]}" --dependency=afterany:$s0 --job-name="cc-eval-35b-${TAG}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1w2=$(sbatch --parsable --time=03:30:00 "${PARTFLAGS[@]}" --dependency=afterany:$s1 --job-name="cc-eval-35b-${TAG}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-35b-${TAG}-agg" \
    --export=ALL,MODE=fcsale,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=910,MAX_ERRORS=12 \
    slurm/cc_eval_agg_shards.sh)
  echo "fcsale $TAG: s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"
}

submit_research () {
  local rb="$FS/outputs/cc_eval_${TAG}_research_thinking_32k_vllm"
  local rcommon="MODEL_PATH=$MP,TAG=$TAG,TP=$EVAL_TP,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
  local r0 r1 r0w2 r1w2 ragg
  r0=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --job-name="cc-eval-35b-res-${TAG}-s0" \
    --export=ALL,$rcommon,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  r1=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --job-name="cc-eval-35b-res-${TAG}-s1" \
    --export=ALL,$rcommon,SHARD_IDX=1,OUTPUT_DIR="$rb/shard_1",SAMPLES_JSONL="$rb/shard_1/samples.jsonl",SUMMARY_JSON="$rb/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  r0w2=$(sbatch --parsable --time=04:00:00 "${PARTFLAGS[@]}" --dependency=afterany:$r0 --job-name="cc-eval-35b-res-${TAG}-s0w2" \
    --export=ALL,$rcommon,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  r1w2=$(sbatch --parsable --time=04:00:00 "${PARTFLAGS[@]}" --dependency=afterany:$r1 --job-name="cc-eval-35b-res-${TAG}-s1w2" \
    --export=ALL,$rcommon,SHARD_IDX=1,OUTPUT_DIR="$rb/shard_1",SAMPLES_JSONL="$rb/shard_1/samples.jsonl",SUMMARY_JSON="$rb/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  ragg=$(sbatch --parsable --dependency=afterany:$r0w2:$r1w2 --job-name="cc-eval-35b-res-${TAG}-agg" \
    --export=ALL,MODE=research,OUTPUT_BASE="$rb",EXPECTED_SAMPLES=320,MAX_ERRORS=20 \
    slurm/cc_eval_agg_shards.sh)
  echo "research $TAG: r0=$r0 r1=$r1 w2=$r0w2/$r1w2 agg=$ragg"
}

case "$WHICH" in
  fcsale)   submit_fcsale ;;
  research) submit_research ;;
  both)     submit_fcsale; submit_research ;;
  *) echo "ERROR: WHICH must be fcsale|research|both, got '$WHICH'"; exit 1 ;;
esac
echo "done submitting $TAG ($WHICH) on pli TP=2"

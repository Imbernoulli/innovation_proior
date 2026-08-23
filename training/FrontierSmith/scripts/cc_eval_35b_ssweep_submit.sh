#!/usr/bin/env bash
# Sharded FCS/ALE + Research eval chains for the LoRA r32 s-knob sweep
# (s=0.2/0.3/0.5 merged models), afterok the sweep merge job.
# OFF-AILAB (user directive): pli H100, TP=2, --gres=gpu:2 per shard,
# account=goedelprover qos=pli-low. Flip with EVAL_PART=ailab EVAL_GRES=gpu:1
# EVAL_TP=1 ACCT_FLAGS= if the pli queue stalls (probe: cc-eval-35b-tp2probe).
#   MERGE_JID=<jid> bash scripts/cc_eval_35b_ssweep_submit.sh
set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
cd "$FS"
: "${MERGE_JID:?need MERGE_JID}"

EVAL_PART="${EVAL_PART:-pli}"
EVAL_GRES="${EVAL_GRES:-gpu:2}"
EVAL_TP="${EVAL_TP:-2}"
PARTFLAGS=(--partition="$EVAL_PART" --gres="$EVAL_GRES")
[ "$EVAL_PART" = "pli" ] && PARTFLAGS+=(--account=goedelprover --qos=pli-low)

for s in 02 03 05; do
  mp="$ROOT/models_sft/lora_q36_35bA3b_clean_nom_r32_s${s}_merged"
  tag="q36_lora_r32_s${s}"
  ob="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"
  common="MODEL_PATH=$mp,TAG=$tag,TP=$EVAL_TP,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
  s0=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$MERGE_JID \
    --job-name="cc-eval-35b-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$MERGE_JID \
    --job-name="cc-eval-35b-${tag}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s0w2=$(sbatch --parsable --time=03:30:00 "${PARTFLAGS[@]}" --dependency=afterany:$s0 \
    --job-name="cc-eval-35b-${tag}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1w2=$(sbatch --parsable --time=03:30:00 "${PARTFLAGS[@]}" --dependency=afterany:$s1 \
    --job-name="cc-eval-35b-${tag}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-35b-${tag}-agg" \
    --export=ALL,MODE=fcsale,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=910,MAX_ERRORS=12 \
    slurm/cc_eval_agg_shards.sh)
  echo "$tag fcsale: s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"

  rb="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
  rcommon="MODEL_PATH=$mp,TAG=$tag,TP=$EVAL_TP,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
  r0=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$MERGE_JID \
    --job-name="cc-eval-35b-res-${tag}-s0" \
    --export=ALL,$rcommon,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  r1=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$MERGE_JID \
    --job-name="cc-eval-35b-res-${tag}-s1" \
    --export=ALL,$rcommon,SHARD_IDX=1,OUTPUT_DIR="$rb/shard_1",SAMPLES_JSONL="$rb/shard_1/samples.jsonl",SUMMARY_JSON="$rb/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  r0w2=$(sbatch --parsable --time=04:00:00 "${PARTFLAGS[@]}" --dependency=afterany:$r0 \
    --job-name="cc-eval-35b-res-${tag}-s0w2" \
    --export=ALL,$rcommon,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  r1w2=$(sbatch --parsable --time=04:00:00 "${PARTFLAGS[@]}" --dependency=afterany:$r1 \
    --job-name="cc-eval-35b-res-${tag}-s1w2" \
    --export=ALL,$rcommon,SHARD_IDX=1,OUTPUT_DIR="$rb/shard_1",SAMPLES_JSONL="$rb/shard_1/samples.jsonl",SUMMARY_JSON="$rb/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  ragg=$(sbatch --parsable --dependency=afterany:$r0w2:$r1w2 --job-name="cc-eval-35b-res-${tag}-agg" \
    --export=ALL,MODE=research,OUTPUT_BASE="$rb",EXPECTED_SAMPLES=320,MAX_ERRORS=20 \
    slurm/cc_eval_agg_shards.sh)
  echo "$tag research: s0=$r0 s1=$r1 w2=$r0w2/$r1w2 agg=$ragg"
done

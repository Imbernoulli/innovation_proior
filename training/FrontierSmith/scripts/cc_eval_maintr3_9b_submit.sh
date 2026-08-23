#!/usr/bin/env bash
# Eval the maintain-r3 9B SFT models (task: does the NEW maintain data help 9B?).
# Reuses the model-agnostic eval batch scripts at TP=1 (9B fits on one GPU),
# 2-way DP sharded FCS/ALE + research, w2 resumes, CPU agg. RAM right-sized to
# 110G (RC email: single-GPU evals must not over-allocate CPU mem).
#   bash scripts/cc_eval_maintr3_9b_submit.sh [fcsale|research|both]
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
MS="$ROOT/models_sft"
cd "$FS"
WHICH="${1:-both}"
MEMFLAG="--mem=110G"

submit_fcsale () {  # tag model_path
  local tag="$1" mp="$2"
  local ob="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=03:00:00 $MEMFLAG --job-name="cc-eval-9b-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1=$(sbatch --parsable --time=03:00:00 $MEMFLAG --job-name="cc-eval-9b-${tag}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s0w2=$(sbatch --parsable --time=02:00:00 $MEMFLAG --dependency=afterany:$s0 --job-name="cc-eval-9b-${tag}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1w2=$(sbatch --parsable --time=02:00:00 $MEMFLAG --dependency=afterany:$s1 --job-name="cc-eval-9b-${tag}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-9b-${tag}-agg" \
    --export=ALL,MODE=fcsale,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=910,MAX_ERRORS=12 \
    slurm/cc_eval_agg_shards.sh)
  echo "fcsale $tag: s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"
}

submit_research () {  # tag model_path
  local tag="$1" mp="$2"
  local ob="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=04:00:00 $MEMFLAG --job-name="cc-eval-9b-res-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  s1=$(sbatch --parsable --time=04:00:00 $MEMFLAG --job-name="cc-eval-9b-res-${tag}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  s0w2=$(sbatch --parsable --time=02:30:00 $MEMFLAG --dependency=afterany:$s0 --job-name="cc-eval-9b-res-${tag}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  s1w2=$(sbatch --parsable --time=02:30:00 $MEMFLAG --dependency=afterany:$s1 --job-name="cc-eval-9b-res-${tag}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-9b-res-${tag}-agg" \
    --export=ALL,MODE=research,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=320,MAX_ERRORS=20 \
    slurm/cc_eval_agg_shards.sh)
  echo "research $tag: s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"
}

MODELS=(
  "maintr3_clean|$MS/sft_q35_clean_maintr3"
  "maintr3_pure|$MS/sft_q35_maintr3_pure"
  "maintr3_filt|$MS/sft_q35_clean_maintr3_filt"
)
for row in "${MODELS[@]}"; do
  IFS='|' read -r tag mp <<<"$row"
  [ -e "$mp/config.json" ] || { echo "SKIP $tag: no config.json at $mp"; continue; }
  case "$WHICH" in
    fcsale)   submit_fcsale "$tag" "$mp" ;;
    research) submit_research "$tag" "$mp" ;;
    both)     submit_fcsale "$tag" "$mp"; submit_research "$tag" "$mp" ;;
  esac
done
echo "done submitting maintr3 9B evals ($WHICH)"

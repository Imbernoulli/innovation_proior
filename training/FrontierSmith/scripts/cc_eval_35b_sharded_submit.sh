#!/usr/bin/env bash
# Submit 2-way data-parallel sharded FCS/ALE + research evals for the 35B line
# (fresh-start TAGs only; resumes with real historical samples stay 1-shard).
# Chain per TAG: s0 + s1 (6h, 1 GPU each) -> s0w2/s1w2 afterany resumes (3.5h/4h)
# -> agg (CPU, merges shard samples last-wins + writes canonical summary.json
# via slurm/cc_eval_agg_shards.sh). Wall-clock ~halves; GPU-hours unchanged.
# Throughput bump for FCS/ALE shards: MAX_NUM_SEQS 64->128 (GDN hybrid: only
# 10/40 layers hold KV; ~60G KV pool on H200 takes far more than 64 seqs),
# CONCURRENCY 64->96, REQUEST_TIMEOUT 2400->3600 (queued requests wait longer
# at higher concurrency). Research keeps CONCURRENCY=4 (evaluator GPU sharing).
set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
MS="$ROOT/models_sft"
cd "$FS"

submit_fcsale () {  # tag model_path
  local tag="$1" mp="$2"
  local ob="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=06:00:00 --job-name="cc-eval-35b-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1=$(sbatch --parsable --time=06:00:00 --job-name="cc-eval-35b-${tag}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s0w2=$(sbatch --parsable --time=03:30:00 --dependency=afterany:$s0 --job-name="cc-eval-35b-${tag}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1w2=$(sbatch --parsable --time=03:30:00 --dependency=afterany:$s1 --job-name="cc-eval-35b-${tag}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-35b-${tag}-agg" \
    --export=ALL,MODE=fcsale,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=910,MAX_ERRORS=12 \
    slurm/cc_eval_agg_shards.sh)
  echo "$tag: s0=$s0 s1=$s1 s0w2=$s0w2 s1w2=$s1w2 agg=$agg"
}

submit_research () {  # tag model_path
  local tag="$1" mp="$2"
  local ob="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=06:00:00 --job-name="cc-eval-35b-res-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  s1=$(sbatch --parsable --time=06:00:00 --job-name="cc-eval-35b-res-${tag}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  s0w2=$(sbatch --parsable --time=04:00:00 --dependency=afterany:$s0 --job-name="cc-eval-35b-res-${tag}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  s1w2=$(sbatch --parsable --time=04:00:00 --dependency=afterany:$s1 --job-name="cc-eval-35b-res-${tag}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    slurm/cc_eval_research_ailab.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-35b-res-${tag}-agg" \
    --export=ALL,MODE=research,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=320,MAX_ERRORS=20 \
    slurm/cc_eval_agg_shards.sh)
  echo "research $tag: s0=$s0 s1=$s1 s0w2=$s0w2 s1w2=$s1w2 agg=$agg"
}

echo "== FCS/ALE sharded (fresh-start TAGs) =="
submit_fcsale q36full_wd01_a5      "$MS/soup_q36_35b_clean_nom_full_wd01_a5"
submit_fcsale q36full_wd01_sft     "$MS/sft_q36_35bA3b_clean_nom_full_wd01"
submit_fcsale q36full_wd03_sft     "$MS/sft_q36_35bA3b_clean_nom_full_wd03"
submit_fcsale q36full_lr2e6_sft    "$MS/sft_q36_35bA3b_clean_nom_full_lr2e6"
submit_fcsale q36full_lr2e6_a5     "$MS/soup_q36_35b_clean_nom_full_lr2e6_a5"
submit_fcsale q36full_lr2e6_a10    "$MS/soup_q36_35b_clean_nom_full_lr2e6_a10"
submit_fcsale q36_lora_r16_s10     "$MS/lora_q36_35bA3b_clean_nom_r16_s10_merged"
submit_fcsale q36_lora_r32_s10     "$MS/lora_q36_35bA3b_clean_nom_r32_s10_merged"
submit_fcsale q36_lora_r32_wd03_s10 "$MS/lora_q36_35bA3b_clean_nom_r32_wd03_s10_merged"

echo "== research sharded =="
submit_research q36_35bA3b_base  "$ROOT/models/Qwen3.6-35B-A3B"
submit_research q36_lora_r32_s01 "$MS/lora_q36_35bA3b_clean_nom_r32_s01_merged"
submit_research q36_lora_r32_s10 "$MS/lora_q36_35bA3b_clean_nom_r32_s10_merged"

#!/usr/bin/env bash
# 64k-budget diagnostic: is the whole RL campaign measuring the 32768 cap?
# 63.4% of eval samples hit the cap; cap-hit samples score ~0 (mean FCS 0.35 vs
# 16.40 for completed), and r(cap-hit%, FCS) = -0.808 over 17 model cells. If FCS
# jumps for every model at 64k and the ordering collapses, the campaign measured
# termination, not capability. No training involved.
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"
MODELS=(
  "base|$FS/models/Qwen3.5-9B-bf16"
  "loraIMs20|$FS/models_rl/rlsy_loraIM_s20_hf"
  "soupNEWs20|$FS/models_rl/rlsy_soupNEW10_s20_hf"
  "soupWDs20|$FS/models_rl/rlsy_soupWD03_20_s20_hf"
)
for row in "${MODELS[@]}"; do
  IFS='|' read -r tag mp <<<"$row"
  [ -e "$mp/config.json" ] || { echo "SKIP $tag"; continue; }
  ob="$FS/outputs/cc_eval_64k_${tag}_fcs"
  common="MODEL_PATH=$mp,TAG=64k_$tag,NUM_SHARDS=2,MAX_NUM_SEQS=64,CONCURRENCY=48,REQUEST_TIMEOUT=7200,GJ_BACKEND=auto,MAX_TOKENS=65536,SOURCE=frontiercs,NUM_SAMPLES=3"
  for sh in 0 1; do
    sbatch --parsable --time=05:00:00 --mem=110G --job-name="cc-64k-${tag}-s${sh}" \
      --export=ALL,$common,SHARD_IDX=$sh,OUTPUT_DIR="$ob/shard_$sh",SAMPLES_JSONL="$ob/shard_$sh/samples.jsonl",SUMMARY_JSON="$ob/shard_$sh/summary_shard.json" \
      slurm/cc_eval_thinking_both_ailab.sh
  done | tr '\n' ' '
  echo "  <- $tag"
done

#!/usr/bin/env bash
# Submit one eval as a CHAIN of short passes on a fast-scheduling QOS.
#
# Why: the shared `gpu` partition quotes a ~9.5 h Priority wait for a 3 h job,
# while gpu-test (priority 8000, 1 h cap, 3 jobs/user) schedules in seconds. The
# eval generator resumes from SAMPLES_JSONL (--resume + _load_existing dedupes on
# problem/sample id), so an eval that needs N hours can be N one-hour passes that
# each start immediately. Total wall time is far lower than waiting for one long
# slot.
#
# Usage: TAG=<tag> MODEL=<hf dir> PASSES=6 scripts/eval_fast_chain.sh [fcsale|research]
set -euo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"
: "${TAG:?}" "${MODEL:?}"
KIND="${1:-fcsale}"
PASSES="${PASSES:-6}"
QOS_FLAGS=(${EVAL_QOS_FLAGS:---qos=gpu-test --gres=gpu:a100:1})
WALL="${EVAL_WALL:-01:00:00}"

# N_SAMPLES tops up an EXISTING eval: --resume keeps every sample already on disk
# and generates only the new sample_idx values, because planned_keys is built from
# range(n_samples). Raising it from 5 to 16 therefore costs 11 new samples/problem,
# not a rerun. (Our per-problem comparisons are sampling-noise limited -- ~91% of
# the paired-difference variance on FCS is sampling noise -- so more samples per
# problem, not more problems, is what buys a trustworthy headline number.)
N_SAMPLES="${N_SAMPLES:-5}"

if [ "$KIND" = "research" ]; then
  script=slurm/cc_eval_research_autopart.sh
  ob="$FS/outputs/cc_eval_${TAG}_research_thinking_32k_vllm"
  extra="CONCURRENCY=32,MAX_NUM_SEQS=64,FRONTIERCS_RESEARCH_MAX_CONC=4,N_SAMPLES=$N_SAMPLES"
else
  script=slurm/cc_eval_thinking_both_autopart.sh
  ob="$FS/outputs/cc_eval_${TAG}_thinking_32k_both_vllm"
  extra="NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600,GJ_BACKEND=auto"
fi

for shard in 0 1; do
  prev=""
  for p in $(seq 1 "$PASSES"); do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    j=$(sbatch --parsable --time="$WALL" --mem=110G "${QOS_FLAGS[@]}" $dep \
      --job-name="cce-${TAG}-s${shard}p${p}" \
      --export=ALL,MODEL_PATH="$MODEL",TAG="$TAG",$extra,SHARD_IDX=$shard,\
OUTPUT_DIR="$ob/shard_$shard",SAMPLES_JSONL="$ob/shard_$shard/samples.jsonl",SUMMARY_JSON="$ob/shard_$shard/summary_shard.json" \
      "$script")
    prev=$j
  done
  echo "  shard $shard: $PASSES 段链，末段 $prev"
done

#!/usr/bin/env bash
# Submit the CPU-only eval clients that talk to the shared vLLM pool.
#
# Architecture: scripts/vllm_pool_serve.sh runs one vLLM per model on GPUs and
# publishes <node>:<port> to a registry on shared GPFS; these clients run the
# judge + eval driver on the `cpu` partition and connect straight to a backend.
# Compute-node-to-compute-node TCP was verified to work, so no login-node proxy
# is involved (it would be a single point of failure and would funnel every
# 32k-token response through one host).
#
# Clients can be submitted BEFORE the backends are ready: cc_eval_cpu_client.sh
# polls the registry for up to POOL_WAIT seconds, so start order does not matter.
#
# Usage: scripts/submit_pool_clients.sh <tag> [tag...]
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"

SHARDS="${SHARDS:-2}"
CPUS="${CPUS:-32}"

for tag in "$@"; do
  ob_f="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"
  ob_r="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
  for shard in $(seq 0 $((SHARDS-1))); do
    # FCS + ALE
    jf=$(sbatch --parsable --job-name="cli-${tag}-f${shard}" -c "$CPUS" \
      --export=ALL,TAG="$tag",MODEL_TAG="$tag",KIND=fcsale,\
SOURCE=both,NUM_SHARDS=$SHARDS,SHARD_IDX=$shard,\
OUTPUT_DIR="$ob_f/shard_$shard",SAMPLES_JSONL="$ob_f/shard_$shard/samples.jsonl",\
SUMMARY_JSON="$ob_f/shard_$shard/summary_shard.json" \
      slurm/cc_eval_cpu_client.sh)
    # Research: no go-judge needed (its evaluator runs in-process), so KIND
    # skips the judge startup entirely.
    jr=$(sbatch --parsable --job-name="cli-${tag}-r${shard}" -c "$CPUS" \
      --export=ALL,TAG="$tag",MODEL_TAG="$tag",KIND=research,\
SOURCE=research,NUM_SHARDS=$SHARDS,SHARD_IDX=$shard,\
OUTPUT_DIR="$ob_r/shard_$shard",SAMPLES_JSONL="$ob_r/shard_$shard/samples.jsonl",\
SUMMARY_JSON="$ob_r/shard_$shard/summary_shard.json" \
      slurm/cc_eval_cpu_client.sh)
    echo "  $tag shard$shard: fcsale=$jf research=$jr"
  done
done

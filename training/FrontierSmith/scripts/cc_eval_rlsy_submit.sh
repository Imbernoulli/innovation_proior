#!/usr/bin/env bash
# Eval RL checkpoints of the rlsy (synth-only multisource) arms: four benches.
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
# 2026-08-14: ailab QOS caps a user at gres/gpu=16 and our 4 training arms hold
# exactly 16, so every eval queued on ailab starves until training ends. Offload
# to partitions the training jobs do not touch (A100 whole-card avoids MIG; the
# research track needs pli H100 per the eval-partition playbook). Override with
# FS_EVAL_PART / FS_RES_PART.
# della rejects an explicit --partition; route by QOS instead (this is how the
# historical gpu-partition evals were submitted: account=chij + gpu-* QOS).
# 2026-08-15: gpu-short/-medium on the shared `gpu` partition quote a ~9.5 h
# Priority wait; a probe on gpu-ee (EE nodes, della-gpuee QOS) started in 0 s.
# Spread across both so whichever frees first runs -- gpu-ee only has 2 cards,
# so it cannot absorb the whole queue on its own.
EVAL_PART=(${FS_EVAL_PART:---qos=gpu-medium --gres=gpu:a100:1})
EVAL_PART_ALT=(${FS_EVAL_PART_ALT:---partition=gpu-ee --qos=della-gpuee --gres=gpu:a100:1})
# --gres=gpu:1 can land on a MIG slice (1g.10gb) where a 9B model does not
# even fit its weights -> vLLM dies in profile_run. Always name the card.
RES_PART=(${FS_RES_PART:---partition=pli --account=goedelprover --qos=pli-low --gres=gpu:h100:1})

submit_fcsale () {  # tag model_path
  local tag="$1" mp="$2"
  local ob="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600,GJ_BACKEND=auto"  # GJ_BACKEND=auto: userns dead -> hybrid starter falls back to gojudge_shim
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=03:00:00 $MEMFLAG --job-name="cc-eval-9b-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    "${EVAL_PART[@]}" slurm/cc_eval_thinking_both_autopart.sh)
  s1=$(sbatch --parsable --time=03:00:00 $MEMFLAG --job-name="cc-eval-9b-${tag}-s1" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    "${EVAL_PART[@]}" slurm/cc_eval_thinking_both_autopart.sh)
  s0w2=$(sbatch --parsable --time=02:00:00 $MEMFLAG --dependency=afterany:$s0 --job-name="cc-eval-9b-${tag}-s0w2" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    "${EVAL_PART[@]}" slurm/cc_eval_thinking_both_autopart.sh)
  s1w2=$(sbatch --parsable --time=02:00:00 $MEMFLAG --dependency=afterany:$s1 --job-name="cc-eval-9b-${tag}-s1w2" \
    --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    "${EVAL_PART[@]}" slurm/cc_eval_thinking_both_autopart.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-9b-${tag}-agg" \
    --export=ALL,MODE=fcsale,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=910,MAX_ERRORS=12 \
    slurm/cc_eval_agg_shards.sh)
  echo "fcsale $tag: s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"
}

submit_research () {  # tag model_path
  local tag="$1" mp="$2"
  local ob="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
  # 2026-08-15: research shards ran at CONCURRENCY=4 (the script default) while
  # FCS/ALE got 96, so 320 samples x ~83 s serialised into >4 h and hit the wall
  # (12386567 TIMEOUT at 228/320). Generation throughput was 377 tok/s vs 1749
  # tok/s/GPU in training. Raise concurrency and give the simulators room.
  local rescommon="CONCURRENCY=${RES_CONC:-32},MAX_NUM_SEQS=${RES_SEQS:-64},FRONTIERCS_RESEARCH_MAX_CONC=${RES_SCORE_CONC:-4}"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=04:00:00 $MEMFLAG --job-name="cc-eval-9b-res-${tag}-s0" \
    --export=ALL,$common,$rescommon,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    "${RES_PART[@]}" slurm/cc_eval_research_autopart.sh)
  s1=$(sbatch --parsable --time=04:00:00 $MEMFLAG --job-name="cc-eval-9b-res-${tag}-s1" \
    --export=ALL,$common,$rescommon,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    "${RES_PART[@]}" slurm/cc_eval_research_autopart.sh)
  s0w2=$(sbatch --parsable --time=02:30:00 $MEMFLAG --dependency=afterany:$s0 --job-name="cc-eval-9b-res-${tag}-s0w2" \
    --export=ALL,$common,$rescommon,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    "${RES_PART[@]}" slurm/cc_eval_research_autopart.sh)
  s1w2=$(sbatch --parsable --time=02:30:00 $MEMFLAG --dependency=afterany:$s1 --job-name="cc-eval-9b-res-${tag}-s1w2" \
    --export=ALL,$common,$rescommon,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" \
    "${RES_PART[@]}" slurm/cc_eval_research_autopart.sh)
  agg=$(sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="cc-eval-9b-res-${tag}-agg" \
    --export=ALL,MODE=research,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=320,MAX_ERRORS=20 \
    slurm/cc_eval_agg_shards.sh)
  echo "research $tag: s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"
}

MODELS=(
  "rlmxW_s5|/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models_rl/rlmx_soupWD03_20_s5_hf"
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
echo "done submitting rlsy RL-ckpt evals ($WHICH)"

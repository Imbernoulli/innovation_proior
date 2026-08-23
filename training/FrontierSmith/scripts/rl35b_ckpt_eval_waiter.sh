#!/usr/bin/env bash
# =============================================================================
# 35B RL checkpoint eval waiter (one per RL arm).
#
# Polls the arm's verl checkpoint dir; for each COMPLETE model-only save
# (global_step_{5,10,15,20}: 8 rank shards, >=55 GiB) it chains, once:
#   export (CPU: merge_fsdp_to_hf, bf16, ~67G HF dir under models_rl/)
#     -> afterok: 2-way sharded FCS/ALE eval OFF-AILAB (pli H100, TP=2,
#        --gres=gpu:2 per shard) + w2 resumes + CPU agg (canonical summary.json)
#     -> for s20 additionally the sharded research eval (pli, TP=2, GMU 0.62).
# Disk discipline: df checked EVERY poll and before every submit chain;
# <500G => ALARM in the status log and the waiter stops submitting (exit 9).
#
# Env: EXP (rl35b_r32s01|rl35b_base), ARMTAG (r32s01|base), START_MODEL
#      (dir to copy processor 四件套/tokenizer from), STEPS_WANTED="5 10 15 20".
# =============================================================================
#SBATCH --job-name=rl35b_evalwaiter
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
set -uo pipefail

ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
cd "$FS"
EXP="${EXP:?}"; ARMTAG="${ARMTAG:?}"; START_MODEL="${START_MODEL:?}"
STEPS_WANTED="${STEPS_WANTED:-5 10 15 20}"
CK="$FS/checkpoints/rl_frontiersmith_synth/$EXP"
STATUS="$FS/logs/rl35b_pipeline_status.log"
EVAL_PART="${EVAL_PART:-pli}"          # off-ailab first (user directive)
EVAL_GRES="${EVAL_GRES:-gpu:2}"        # 35B on 80G H100 needs TP=2
EVAL_TP="${EVAL_TP:-2}"
POLL="${POLL:-600}"
# pli lives under a different account/QOS pair (plain sbatch -> "Invalid qos").
PARTFLAGS=(--partition="$EVAL_PART" --gres="$EVAL_GRES")
if [ "$EVAL_PART" = "pli" ]; then
  PARTFLAGS+=(--account=goedelprover --qos=pli-low)
fi
log() { echo "[$(date '+%F %T')] [waiter-$ARMTAG] $*" | tee -a "$STATUS"; }

ckpt_complete() {  # step -> 0/1
  local a="$CK/global_step_$1/actor"
  [ -d "$a" ] || return 1
  local n sz
  n=$(ls "$a"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
  sz=$(du -s --apparent-size --block-size=1G "$CK/global_step_$1" 2>/dev/null | cut -f1)
  [ "$n" -eq 8 ] && [ "${sz:-0}" -ge 55 ]
}

submit_chain() {  # step
  local step="$1"
  local tag="${EXP}_s${step}"
  local hf="$ROOT/models_rl/${tag}_hf"
  local ob="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"

  # export (CPU)
  local ej
  ej=$(sbatch --parsable --job-name="rl35b_export_${ARMTAG}_s${step}" \
      --partition=cpu --cpus-per-task=8 --mem=400G --time=04:00:00 \
      --export=ALL,CKPT_PATH="$CK/global_step_${step}",OUTPUT_DIR="$hf" \
      slurm/export_verl_ckpt_to_hf_cpu.sh)
  # processor 四件套 + tokenizer completion (idempotent cp -n) as a tiny afterok job
  local pj
  pj=$(sbatch --parsable --job-name="rl35b_proc4_${ARMTAG}_s${step}" --dependency=afterok:$ej \
      --partition=cpu --cpus-per-task=1 --mem=2G --time=00:10:00 \
      --wrap="for f in preprocessor_config.json processor_config.json video_preprocessor_config.json chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do cp -n '$START_MODEL'/\$f '$hf'/ 2>/dev/null; done; ls '$hf'/config.json")

  # sharded FCS/ALE on pli TP=2
  local common="MODEL_PATH=$hf,TAG=$tag,TP=$EVAL_TP,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
  local s0 s1 s0w2 s1w2 agg
  s0=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$pj \
    --job-name="cc-eval-35b-${tag}-s0" \
    --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" \
    slurm/cc_eval_thinking_both_ailab.sh)
  s1=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$pj \
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
  log "s${step}: export=$ej proc4=$pj fcsale s0=$s0 s1=$s1 w2=$s0w2/$s1w2 agg=$agg"

  # research for s20
  if [ "$step" = "20" ]; then
    local rb="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
    local rcommon="MODEL_PATH=$hf,TAG=$tag,TP=$EVAL_TP,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
    local r0 r1 r0w2 r1w2 ragg
    r0=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$pj \
      --job-name="cc-eval-35b-res-${tag}-s0" \
      --export=ALL,$rcommon,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" \
      slurm/cc_eval_research_ailab.sh)
    r1=$(sbatch --parsable --time=06:00:00 "${PARTFLAGS[@]}" --dependency=afterok:$pj \
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
    log "s${step}: research r0=$r0 r1=$r1 w2=$r0w2/$r1w2 agg=$ragg"
  fi
}

log "started for $EXP (steps: $STEPS_WANTED; evals on $EVAL_PART $EVAL_GRES TP=$EVAL_TP)"
end=$((SECONDS + 47*3600))
while [ $SECONDS -lt $end ]; do
  freeg=$(df -BG --output=avail /scratch/gpfs/CHIJ | tail -1 | tr -dc 0-9)
  if [ "${freeg:-0}" -lt 500 ]; then
    log "ALARM: disk ${freeg}G < 500G -- STOPPING all further submissions. Human needed."
    exit 9
  fi
  remaining=0
  for s in $STEPS_WANTED; do
    m="$CK/.evalchain_s${s}_submitted"
    [ -e "$m" ] && continue
    if ckpt_complete "$s"; then
      submit_chain "$s" && touch "$m"
    else
      remaining=1
    fi
  done
  [ "$remaining" = 0 ] && { log "all steps ($STEPS_WANTED) chained -- waiter done."; exit 0; }
  sleep "$POLL"
done
log "waiter walltime reached with steps still pending -- resubmit me if the run is still going."
exit 0

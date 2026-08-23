#!/usr/bin/env bash
# =============================================================================
# rlv4_ckpt_eval_waiter.sh -- one waiter per rlv4 arm. Polls the arm's checkpoint
# dir; as soon as a wanted step lands COMPLETE, it
#   1. exports FSDP shards -> HF bf16 (CPU job, models_rl/rlv4_<arm>_s<N>_hf)
#   2. copies the processor/tokenizer 四件套 from the arm's START model
#   3. submits the four-bench eval chain (FCS+ALE sharded, research sharded)
#      by rewriting the MODELS line of scripts/cc_eval_rlsy_submit.sh into a
#      per-step throwaway copy (the shared script is never edited in place).
# Adapted from rl35b_ckpt_eval_waiter.sh; 9B/TP=1/4-GPU-world differences applied.
#
# Env: ARM (base|loraIM|soupNEW10|soupWD03_20), START_MODEL (dir with the
#      tokenizer/processor files), STEPS_WANTED="5 10 15 20", WORLD (4),
#      POLL (600s), MIN_CKPT_GB (25).
# =============================================================================
#SBATCH --job-name=rlv4_evalwaiter
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

ARM="${ARM:?set ARM=base|loraIM|soupNEW10|soupWD03_20}"
START_MODEL="${START_MODEL:?set START_MODEL to the starting model dir of this arm}"
STEPS_WANTED="${STEPS_WANTED:-5 10 15 20}"
WORLD="${WORLD:-4}"                 # rlv4 trains on 4 GPUs -> 4 FSDP shards
MIN_CKPT_GB="${MIN_CKPT_GB:-25}"    # 9B actor+optim shards ~36G; 25 is a safe floor
POLL="${POLL:-600}"

EXP="${EXP_PREFIX:-rlv6}_${ARM}"
CK="$FS/checkpoints/rl_multisource/$EXP"
STATUS="$FS/logs/rlv6_pipeline_status.log"

log() { echo "[$(date '+%F %T')] [waiter-$ARM] $*" | tee -a "$STATUS"; }

ckpt_complete() {  # step -> 0/1
  local a="$CK/global_step_$1/actor"
  [ -d "$a" ] || return 1
  local n sz
  n=$(ls "$a"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
  sz=$(du -s --apparent-size --block-size=1G "$CK/global_step_$1" 2>/dev/null | cut -f1)
  [ "$n" -eq "$WORLD" ] && [ "${sz:-0}" -ge "$MIN_CKPT_GB" ]
}

submit_chain() {  # step
  local step="$1"
  local tag="${EXP}_s${step}"
  local hf="$ROOT/models_rl/${tag}_hf"
  local ck="$CK/global_step_$step/actor"
  local j
  # One CPU job does export -> processor copy -> four-bench submission, so the
  # eval chain can never race ahead of an incomplete HF dir.
  j=$(sbatch --parsable --job-name="rlv6_expeval_${ARM}_s${step}" \
      --partition=cpu --cpus-per-task=8 --mem=200G --time=04:00:00 \
      --output="$FS/logs/%x-%j.out" --error="$FS/logs/%x-%j.err" \
      --export=ALL,CKPT_DIR="$ck",HF_OUT="$hf",START_MODEL="$START_MODEL",TAG="$tag" \
      "$FS/scripts/rlv4_export_and_eval.sh")
  log "s$step: export+eval job $j -> $hf"
}

log "start: exp=$EXP ck=$CK steps=[$STEPS_WANTED] world=$WORLD poll=${POLL}s"
declare -A DONE=()
while :; do
  all_done=1
  for st in $STEPS_WANTED; do
    [ "${DONE[$st]:-0}" = "1" ] && continue
    all_done=0
    if ckpt_complete "$st"; then
      log "s$st: checkpoint COMPLETE -> launching export+eval chain"
      submit_chain "$st"
      DONE[$st]=1
    fi
  done
  [ "$all_done" = "1" ] && { log "all wanted steps handled -- exiting"; break; }
  sleep "$POLL"
done

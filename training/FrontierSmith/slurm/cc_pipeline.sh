#!/usr/bin/env bash
# CONSOLIDATED auto-submit pipeline: Instruct -> SFT -> average(soup) -> RL, with FCS/ALE+MLS eval at
# every stage. Dependency-chained (slurm afterok; NO polling loops). Idempotent (skip-if-exists).
# Every known-bug fix baked in: tools/system/loss-normalized data, proc-config copy for vLLM, 182-scope
# eval, MLS-devfix served-name, RL save-safe config, verl->HF export before RL eval.
#
#   bash cc_pipeline.sh <variant>            # variant in {methodv4_r2, methodtraj_v4_r2, method_r2, methodtraj_r2}
#   SOUP_ALPHA=0.3 bash cc_pipeline.sh ...   # soup weight on SFT (default 0.3 = the a30 that amplified under RL)
set -uo pipefail
V="${1:?need variant}"
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; LF=$ROOT/LF-innov
START=$FS/models/Qwen3.5-9B-bf16; ENV=$ROOT/envs/sft_lf
A="${SOUP_ALPHA:-0.3}"; APCT=$(python3 -c "print(int(round($A*100)))")
SFT=$ROOT/models_sft/sft_q35_a100_$V
SOUP=$ROOT/models_sft/soup_q35_a100_${V}_a${APCT}
RLTAG=rl_${V}_a${APCT}
RLCK=$FS/checkpoints/rlm_amplify_v3/$RLTAG
HF=$ROOT/models/${RLTAG}_step40_hf
TRAIN=$FS/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet
SOUP_PY=$FS/scripts/cc_model_soup_merge.py
PROC="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"
mkdir -p $FS/logs
MAN=$FS/logs/pipeline_${V}.manifest
echo "=== pipeline $V (soup a$APCT) $(date) ===" >>$MAN
have_full(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
log(){ echo "  $*"; echo "$*" >>$MAN; }

# ---- 1) SFT (Instruct -> full-FT) ----
if have_full "$SFT"; then JSFT=""; log "[skip] SFT exists: $SFT"
else
  JSFT=$(cd $LF && sbatch --parsable -J sft_$V --gres=gpu:4 --cpus-per-task=32 --mem=480G --time=06:00:00 \
        cc-sft-innov.sh examples/train_full/auto/os-q35_a100_$V.yaml)
  log "SFT job=$JSFT"
fi
DSFT=${JSFT:+--dependency=afterok:$JSFT}

# ---- 2) soup-average (CPU) after SFT ----
if have_full "$SOUP"; then JM=""; log "[skip] soup exists: $SOUP"
else
  JM=$(sbatch --parsable -J soup_${V}_a$APCT --partition=cpu --cpus-per-task=8 --mem=200G --time=01:30:00 \
       -o $FS/logs/%x-%j.out -e $FS/logs/%x-%j.err $DSFT \
       --wrap "$ENV/bin/python $SOUP_PY --sft $SFT --base $START --alpha $A --out $SOUP && $PROC $SOUP/ && $PROC $SFT/")
  log "soup job=$JM"
fi
DM=${JM:+--dependency=afterok:$JM}

# ---- 3) eval raw-SFT (after SFT) + soup (after merge): FCS+ALE 32k (182) + MLS-devfix ----
ev(){ # dir tag dep
  local dir="$1" tag="$2" dep="$3"
  [ -d "$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm" ] || \
    sbatch --parsable -J cc-eval-$tag $dep --time=06:00:00 --export=ALL,MODEL_PATH=$dir,TAG=$tag \
      $FS/slurm/cc_eval_thinking_both_ailab.sh >>$MAN 2>&1 && log "FCS+ALE eval $tag"
  [ -d "$FS/outputs/cc_mlsbench_cpu_${tag}_devfix" ] || \
    (cd $FS && sbatch --parsable -J cc-mlsdevfix-$tag $dep --time=08:00:00 \
      --export=ALL,MODEL_PATH=$dir,TAG=${tag}_devfix,MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev,EVAL_RESEARCHER_YEAR=2026 \
      slurm/cc_eval_mlsbench_cpu_ailab.sh) >>$MAN 2>&1 && log "MLS eval $tag"
}
ev "$SFT"  "sftr2_$V"          "$DSFT"
ev "$SOUP" "soupr2_${V}_a$APCT" "$DM"

# ---- 4) RL on the soup (after merge) ----
if [ -d "$RLCK/global_step_20" ]; then JRL=""; log "[skip] RL done: $RLCK"
else
  JRL=$(cd $FS && sbatch --parsable -J rlm3_$RLTAG --gres=gpu:4 --cpus-per-task=32 --mem=480G --time=24:00:00 $DM \
    --export=ALL,MODEL_PATH=$SOUP,TRAIN_DATA=$TRAIN,EXPERIMENT_NAME=rlm3_$RLTAG,PROJECT_NAME=rlm_amplify_v3,CKPT_DIR=$RLCK,ROLLOUT_DIR=$FS/outputs/rlm_amplify_v3_rollout/$RLTAG,FS_PERTASK_REWARD_NORM=1,TOTAL_TRAINING_STEPS=20,SAVE_FREQ=5,MAX_ACTOR_CKPT_TO_KEEP=10,TEST_FREQ=100000,FRONTIERCS_JUDGE_MAX_WAIT=900,FRONTIERCS_JUDGE_FAIL_SOFT=1,NGPU=4 \
    slurm/rlm_amplify_v3_ailab.sh)
  log "RL job=$JRL"
fi
DRL=${JRL:+--dependency=afterok:$JRL}

# ---- 5) RL export verl->HF (after RL) then matched FCS+ALE + MLS (after export) ----
if have_full "$HF"; then JEX=""; log "[skip] HF export exists: $HF"
else
  JEX=$(sbatch --parsable -J cc-rlexport-$RLTAG --gres=gpu:2 --cpus-per-task=16 --mem=400G --time=12:00:00 $DRL \
    --export=ALL,CKPT_PATH=$RLCK/global_step_20,MODEL_OUTPUT_DIR=$HF,OUTPUT_DIR=$FS/outputs/eval_${RLTAG}_step40_vllm,SERVED_MODEL_NAME=$RLTAG,VLLM_PORT=8000,PORT=8082,GJ_PORT=5050 \
    $FS/slurm/export_and_eval_qwen35_ckpt_vllm_ailab.sh)
  log "RL export+eval job=$JEX"
fi
log "RL-after MATCHED 32k eval (FCS+ALE+MLS on $HF) is submitted by the watchdog once the HF export lands (it copies proc configs first) — keeps this chain free of the 26624-vs-32k mismatch."
echo "=== pipeline $V submitted; manifest: $MAN ==="

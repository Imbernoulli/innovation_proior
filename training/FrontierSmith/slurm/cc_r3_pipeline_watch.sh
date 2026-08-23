#!/usr/bin/env bash
# Watchdog: when each r3 SFT model lands -> soup(base+SFT a30) -> eval(SFT+soup, strip口径) -> RL(soup, 32k).
# Idempotent (skip-if-exists). Mirrors cc_pipeline but triggered on model appearance (no re-SFT).
set -uo pipefail
# SINGLETON guard: only one instance ever runs (prevents double soup/RL submits).
exec 200>/tmp/cc_r3_watchdog.lock; flock -n 200 || { echo "another watchdog holds the lock; exiting"; exit 0; }
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; ENV=$ROOT/envs/sft_lf
START=$FS/models/Qwen3.5-9B-bf16; SOUP_PY=$FS/scripts/cc_model_soup_merge.py; ELS=$FS/slurm/cc_eval_thinking_both_ailab.sh
TRAIN=$FS/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet
PROC="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"
declare -A SFT=( [methodv4_r3]=$ROOT/models_sft/sft_q35_a100_methodv4_r3 [methodtraj_v4_r3]=$ROOT/models_sft/sft_q35_a100_methodtraj_v4_r3 [full_r3_wd0]=$ROOT/models_sft/sft_q35_full_r3_wd0 [full_r3_wd01]=$ROOT/models_sft/sft_q35_full_r3_wd01 )
have(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
ev(){ local d=$1 t=$2 dep=$3
  [ -d "$FS/outputs/cc_eval_${t}_thinking_32k_both_vllm" ] && return 0          # already run
  squeue -u $USER -h -n cc-eval-$t -o %T 2>/dev/null | grep -q . && return 0    # already queued/running -> DON'T resubmit
  sbatch -J cc-eval-$t $dep --time=06:00:00 --export=ALL,MODEL_PATH=$d,TAG=$t,CONCURRENCY=16,FRONTIERCS_STRIP_THINK_EXTRACT=1 $ELS >/dev/null 2>&1 && echo "  eval $t"; }
for i in $(seq 1 240); do
  allok=1
  for v in "${!SFT[@]}"; do
    S=${SFT[$v]}; SOUP=$ROOT/models_sft/soup_q35_${v}_a30; RLCK=$FS/checkpoints/rlm_amplify_v3/rl_${v}_a30
    if ! have "$S"; then allok=0; continue; fi
    $PROC "$S"/ 2>/dev/null
    # soup
    JM=""
    if ! have "$SOUP" && ! squeue -u $USER -h -n soup_$v -o %T 2>/dev/null | grep -q .; then
      JM=$(sbatch --parsable -J soup_$v --partition=cpu --cpus-per-task=8 --mem=200G --time=01:30:00 -o $FS/logs/%x-%j.out \
        --wrap "$ENV/bin/python $SOUP_PY --sft $S --base $START --alpha 0.3 --out $SOUP && $PROC $SOUP/" 2>/dev/null)
      echo "  [$v] soup job=$JM"
    fi
    DM=${JM:+--dependency=afterok:$JM}
    ev "$S" "r3_sft_$v" ""
    ev "$SOUP" "r3_soup_${v}_a30" "$DM"
    # RL on soup (32k)
    if [ ! -d "$RLCK/global_step_20" ] && ! squeue -u $USER -h -n rlm3_rl_${v}_a30 -o %T|grep -q .; then
      (cd $FS && sbatch -J rlm3_rl_${v}_a30 --gres=gpu:4 --cpus-per-task=32 --mem=480G --time=24:00:00 $DM \
        --export=ALL,MODEL_PATH=$SOUP,TRAIN_DATA=$TRAIN,EXPERIMENT_NAME=rlm3_rl_${v}_a30,PROJECT_NAME=rlm_amplify_v3,CKPT_DIR=$RLCK,ROLLOUT_DIR=$FS/outputs/rlm_amplify_v3_rollout/rl_${v}_a30,MAX_RESPONSE_LENGTH=32768,MAX_MODEL_LEN=45056,MAX_NUM_BATCHED_TOKENS=45056,ROLLOUT_N=4,FS_PERTASK_REWARD_NORM=1,TOTAL_TRAINING_STEPS=20,SAVE_FREQ=5,MAX_ACTOR_CKPT_TO_KEEP=10,TEST_FREQ=100000,NGPU=4 \
        slurm/rlm_amplify_v3_ailab.sh) >/dev/null 2>&1 && echo "  [$v] RL(soup,32k) submitted"
    fi
  done
  [ "$allok" = 1 ] && echo "$(date +%H:%M) all landed SFT chained (keep looping for late arrivals)"
  sleep 300
done

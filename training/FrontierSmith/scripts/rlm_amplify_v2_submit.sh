#!/usr/bin/env bash
# Submit all 4 rlm_amplify_v2 GRPO runs (one per starting model) on the proven
# 4-GPU single-node harness. 4 runs x 4 GPU = 16 GPU = the quota.
#
# The "amplify" experiment: RL on 4 starting models to test whether RL amplifies the
# method-soup more than the start model.
#   base     -> Qwen3.5-9B base CHAT build (control)
#   sfta00   -> pure method-SFT on base
#   soupa00  -> method soup, a00 mix
#   soupa100 -> method soup, a100 mix
#
# Usage:  bash scripts/rlm_amplify_v2_submit.sh
set -euo pipefail

ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
SCRIPT="$FS/slurm/rlm_amplify_v2_4gpu_ailab.sh"
TRAIN="$FS/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet"

# tag  ->  model path  (verified: config.json + safetensors + tokenizer present)
declare -A MODELS=(
  [rlm2_base]="$FS/models/rlm_Qwen3.5-9B-Base-chat"
  [rlm2_sfta00]="$ROOT/models_sft/sft_q35m_a00_innovonly"
  [rlm2_soupa00]="$ROOT/models_sft/soup_q35m_a00_innovonly_soupa50"
  [rlm2_soupa100]="$ROOT/models_sft/soup_q35m_a100_innovonly_soupa50"
)

cd "$FS"
for tag in rlm2_base rlm2_sfta00 rlm2_soupa00 rlm2_soupa100; do
  mp="${MODELS[$tag]}"
  if [ ! -f "$mp/config.json" ]; then
    echo "ABORT: model missing for $tag: $mp" >&2; exit 1
  fi
  ck="$FS/checkpoints/rlm_amplify_v2/$tag"
  ro="$FS/outputs/rlm_amplify_v2_rollout/$tag"
  jid=$(sbatch --parsable \
    --job-name="$tag" \
    --export=ALL,MODEL_PATH="$mp",TRAIN_DATA="$TRAIN",EXPERIMENT_NAME="$tag",PROJECT_NAME=rlm_amplify_v2,CKPT_DIR="$ck",ROLLOUT_DIR="$ro",FS_PERTASK_REWARD_NORM=1,TOTAL_TRAINING_STEPS=40,SAVE_FREQ=2,TEST_FREQ=100000,FRONTIERCS_JUDGE_MAX_WAIT=900,FRONTIERCS_JUDGE_FAIL_SOFT=1 \
    "$SCRIPT")
  echo "submitted $tag -> job $jid  (model=$mp)"
done

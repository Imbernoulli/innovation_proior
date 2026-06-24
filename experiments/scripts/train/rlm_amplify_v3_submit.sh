#!/usr/bin/env bash
# Submit the EXPANDED rlm_amplify_v3 GRPO set on the FAST config
# (slurm/rlm_amplify_v3_ailab.sh: actor offload off, response 20k, micro-batch 2,
# gpu_mem 0.45, save_freq=5, NVLS guard). Runs queue under the 16-GPU ailab cap.
#
# Expansion (user: "3 instruct, 3.5 instruct/base 都要跑，然后好的 sft / sft+avg 也要跑"):
#   START controls (RL on the raw start model):
#     q3_inst_start    q3 instruct          FrontierSmith/models/Qwen3-8B            (text-only Qwen3)
#     q35_inst_start   q35 instruct (bf16)  FrontierSmith/models/Qwen3.5-9B-bf16
#     q35_base_start   q35 base chat        FrontierSmith/models/rlm_Qwen3.5-9B-Base-chat
#   GOOD SFT:
#     q35_sft          models_sft/sft_q35_a100_method
#     q3_sft           models_sft/sft_q3_a100_method
#   GOOD SOUP:
#     q35_soup10       models_sft/soup_q35_a100_method_soupa10
#     q3_soup10        models_sft/soup_q3_a100_method_soupa10
#     q35_base_soup20  models_sft/soup_q35_a00_method_soupa20
#
# GPU layout: pass GPUS=4 (default) or GPUS=2. q3 = Qwen3ForCausalLM text-only (NO
# processor configs). q35 = Qwen3_5ForConditionalGeneration (needs the 3 processor
# configs + chat_template; all target dirs already have them -- verified).
#
# Usage:  bash scripts/rlm_amplify_v3_submit.sh           # all 8, 4 GPU
#         GPUS=2 bash scripts/rlm_amplify_v3_submit.sh    # all 8, 2 GPU
#         ONLY="q35_base_start q3_sft" bash scripts/rlm_amplify_v3_submit.sh
set -euo pipefail

ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
SCRIPT="$FS/slurm/rlm_amplify_v3_ailab.sh"
TRAIN="$FS/data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet"
GPUS="${GPUS:-4}"
STEPS="${STEPS:-40}"
SAVE="${SAVE:-5}"

declare -A MODELS=(
  [q3_inst_start]="$FS/models/Qwen3-8B"
  [q35_inst_start]="$FS/models/Qwen3.5-9B-bf16"
  [q35_base_start]="$FS/models/rlm_Qwen3.5-9B-Base-chat"
  [q35_sft]="$ROOT/models_sft/sft_q35_a100_method"
  [q3_sft]="$ROOT/models_sft/sft_q3_a100_method"
  [q35_soup10]="$ROOT/models_sft/soup_q35_a100_method_soupa10"
  [q3_soup10]="$ROOT/models_sft/soup_q3_a100_method_soupa10"
  [q35_base_soup20]="$ROOT/models_sft/soup_q35_a00_method_soupa20"
)
ORDER=(q35_sft q3_sft q35_soup10 q3_soup10 q35_base_soup20 q35_inst_start q3_inst_start q35_base_start)
[ -n "${ONLY:-}" ] && ORDER=($ONLY)

cd "$FS"
for tag in "${ORDER[@]}"; do
  mp="${MODELS[$tag]}"
  if [ ! -f "$mp/config.json" ]; then
    echo "ABORT: model missing for $tag: $mp" >&2; exit 1
  fi
  # q35 (multimodal arch) needs processor configs; verify before submit.
  arch=$(python3 -c "import json;print(json.load(open('$mp/config.json'))['architectures'][0])" 2>/dev/null || echo "?")
  if [[ "$arch" == *ForConditionalGeneration* ]]; then
    for f in preprocessor_config.json processor_config.json video_preprocessor_config.json chat_template.jinja; do
      if [ ! -f "$mp/$f" ]; then
        echo "  [fix] $tag missing $f -> copying from models/Qwen3.5-9B-bf16"
        cp -n "$FS/models/Qwen3.5-9B-bf16/$f" "$mp/$f"
      fi
    done
  fi
  ck="$FS/checkpoints/rlm_amplify_v3/$tag"
  ro="$FS/outputs/rlm_amplify_v3_rollout/$tag"
  jid=$(sbatch --parsable \
    --job-name="rlm3_$tag" \
    --gres="gpu:$GPUS" \
    --export=ALL,MODEL_PATH="$mp",TRAIN_DATA="$TRAIN",EXPERIMENT_NAME="rlm3_$tag",PROJECT_NAME=rlm_amplify_v3,CKPT_DIR="$ck",ROLLOUT_DIR="$ro",FS_PERTASK_REWARD_NORM=1,TOTAL_TRAINING_STEPS="$STEPS",SAVE_FREQ="$SAVE",TEST_FREQ=100000,FRONTIERCS_JUDGE_MAX_WAIT=900,FRONTIERCS_JUDGE_FAIL_SOFT=1,NGPU="$GPUS" \
    "$SCRIPT")
  echo "submitted rlm3_$tag -> job $jid  (model=$mp, ${GPUS}GPU, arch=$arch)"
done

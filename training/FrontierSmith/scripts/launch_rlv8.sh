#!/usr/bin/env bash
# rlv8: the campaign configuration agreed with the user on 2026-08-12.
#   * batch 64 x 16                       (matches the July rlsy baseline)
#   * in-wave deepening, ceiling 32       (deepen a group the moment it comes
#                                          back all-wrong; extras generate
#                                          alongside the groups still in flight)
#   * overlong penalty 4096 @ 0.5
#   * MLS mixed in, <=1 test per episode, 3000 s cap
#   * rollout gets the whole card: actor param+optimizer offloaded, GMU 0.90,
#     MAX_NUM_SEQS 128
# Three chained segments per arm (23:59 each) cover the ~20-step run.
set -euo pipefail
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

TAG="${TAG:-rlv8}"
STEPS="${STEPS:-20}"
SEGMENTS="${SEGMENTS:-3}"

COMMON="TOTAL_TRAINING_STEPS=$STEPS,TRAIN_BATCH_SIZE=64,PPO_MINI_BATCH_SIZE=16,ROLLOUT_N=16,SAVE_FREQ=5,\
ADAPTIVE_N_ENABLE=1,ADAPTIVE_N_INWAVE=1,ADAPTIVE_N_OVERLAP=0,ADAPTIVE_N_MAX=32,ADAPTIVE_N_MAX_EXTRA=512,\
ADAPTIVE_N_MAX_EXTRA_PER_WORKER=256,ADAPTIVE_N_AGENTS=single_turn_agent,\
FS_OVERLONG_PENALTY=1,FS_OVERLONG_PENALTY_FACTOR=0.15,\
FS_OVERLONG_FILTER=0,LOSS_AGG_MODE=seq-mean-token-mean,\
MAX_NUM_SEQS=128,GPU_MEMORY_UTILIZATION=0.90,ACTOR_PARAM_OFFLOAD=True,ACTOR_OPTIMIZER_OFFLOAD=True,\
FRONTIERSMITH_SYNTH_MAX_CONC=4,FSX_CHILD_MEM_MB=8192,MLS_RL_EPISODE_MEM_MB=16384,\
MLS_RL_MAX_TESTS=1,MLS_RL_EPISODE_TIMEOUT=3000,\
MAX_PROMPT_LENGTH=26624,MAX_MODEL_LEN=59392,\
TRAIN_DATA=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data/multisource_rl/train.parquet"

declare -A M=(
  [base]=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B-bf16
  [loraIM]=/scratch/gpfs/CHIJ/bohan/fs/models_sft/lora_q35_im_r32_s01_merged
  [soupNEW10]=/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_innnew_ft_a10
  [soupWD03_20]=/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_innnew_wd03_ft_a20
)

for arm in ${ARMS_ONLY:-base loraIM soupNEW10 soupWD03_20}; do
  prev=""
  for seg in $(seq 1 "$SEGMENTS"); do
    name="$TAG-$arm"; [ "$seg" -gt 1 ] && name="$TAG-$arm-c$seg"
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    j=$(sbatch --parsable --job-name="$name" --mem=650G --time=23:59:00 $dep \
      --export=ALL,EXPERIMENT_NAME=${TAG}_$arm,MODEL_PATH=${M[$arm]},$COMMON \
      slurm/cc_rl_multisource.sh)
    prev=$j
    echo -n "$j "
  done
  echo "<- $TAG-$arm"
done

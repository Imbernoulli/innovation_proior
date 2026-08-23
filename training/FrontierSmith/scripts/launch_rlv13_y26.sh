#!/usr/bin/env bash
# rlv13: the time-conditioning closure experiment (user directive, 2026-08-18).
#
# Every stage now conditions on time consistently for the first time:
#   * SFT     : each record's system prompt carries the method's REAL historical
#               year ("It is now year 1952..." for Huffman, etc.) -- verified
#               across all 2590 corpus records.
#   * RL      : THIS run. train_y26.parquet prepends "It is now year 2026. You
#               are a good researcher." to every prompt (1468 rows verified;
#               MLS rows keep their tool definitions after the prefix). rlv12
#               trained with NO system prompt, which left the policy
#               prompt-insensitive and made the y26 eval protocol off-policy.
#   * eval    : y26 protocol (EVAL_RESEARCHER_YEAR=2026 + full template),
#               already the default in slurm/cc_eval_pool_selfhosted.sh.
#
# Two arms only, chosen per the user's instruction (base + our best model):
#   base      -- the reference: does time-conditioned RL help even without an
#                innovation prior?
#   soupNEW10 -- our best arm: the one defensible win of the campaign
#                (ALE-40 n=15: +87.4 vs base RL, t-p=3e-5, sign-test p=0.047,
#                excl-top-2 p=1.7e-4), plus directional MLS/Research advantages.
#
# Everything else matches rlv12's winning recipe exactly (penalty 0.15 +
# seq-mean-token-mean, no filter, 64x16, in-wave deepening cap 32), so the ONLY
# difference vs rlv12 is the y26 prefix in rollouts -- rlv13 vs rlv12 is a clean
# A/B on "does conditioning RL on the present close the loop".
set -euo pipefail
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

TAG="${TAG:-rlv13}"
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
TRAIN_DATA=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data/multisource_rl/train_y26.parquet"

declare -A M=(
  [base]=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B-bf16
  [soupNEW10]=/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_innnew_ft_a10
)

for arm in ${ARMS_ONLY:-base soupNEW10}; do
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

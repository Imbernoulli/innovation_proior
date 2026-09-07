#!/usr/bin/env bash
# launch_rlv5_4b.sh -- the 4B twin of launch_rlv5.sh.
#
# The COMMON block below is copied from launch_rlv5.sh character-for-character. That
# is deliberate: the whole point of the 4B run is a controlled 9B-vs-4B comparison,
# and any recipe drift would confound it. Only the model map changes.
#
# WHICH ARMS, AND WHY NOT ALL FOUR
# The 9B campaign ran four arms; at 4B we have room for three (4 GPUs each against a
# self-imposed 14-GPU ceiling), so ft03nm is dropped. That is not arbitrary -- at 9B
# ft03nm_s20 lost to the control arm on all three of FrontierCS (-0.34), research
# (-0.40) and ALE-40 (-18.5). The two kept arms are the two that won something:
# lo32nm led FrontierCS mean@5 (9.31 vs the control's 8.44) and ft01mix led research
# (15.03 vs 12.64, the campaign's only comparison to clear P(>0)=0.95). `base` stays
# because without the control arm "our arms are better" has nothing to be better than.
#
# ON OFFLOAD -- read before "optimising"
# ACTOR_PARAM_OFFLOAD / ACTOR_OPTIMIZER_OFFLOAD stay True and GPU_MEMORY_UTILIZATION
# stays 0.90, exactly as at 9B. It is tempting to drop offload at 4B: sharded over 4
# ranks the actor is ~2.3G of params and ~14G of optimiser state, which looks like it
# fits. It does not fit at GMU=0.90 -- vLLM alone claims 0.90 x 141G = 127G, and
# 127 + 16 > 141. Dropping offload therefore also means dropping GMU to ~0.80, i.e.
# two coupled changes to a recipe whose comment in launch_rlv5.sh already warns that
# the pairing is load-bearing. The step time here is dominated by rollout (agent
# episodes, MLS_RL_EPISODE_TIMEOUT=3000) rather than by the actor update, so the
# expected win is small and the risk of losing a 4-GPU day is not. Measure first:
# the per-phase timings are in the run log, and if the update phase turns out to be a
# large share, revisit with GMU=0.80 as a paired change.
#
# WHY TMPDIR IS IN THE EXPORT LIST
# .bashrc sets TMPDIR=$HOME/.tmp and sbatch --export=ALL carries it onto the node, so
# every tempfile.mkdtemp() in the reward path lands on /home, which has a 50G quota.
# The first 4B wave (13537888/91/94) died 3-5h in with
#   OSError: [Errno 122] Disk quota exceeded: '/home/zy7019/.tmp/fsx_gc_...'
# raised from frontiersmith_synth._score_gen_checker -> tempfile.mkdtemp. The checker
# makes a fresh work dir per scored sample and 64 prompts x 16 rollouts is ~1k dirs a
# step, so home fills whatever headroom it starts with. Repointing TMPDIR at GPFS is
# the whole fix; it is infrastructure, not a recipe parameter, so it does not break
# the "copy the 9B COMMON verbatim" rule.
set -euo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
cd "$D/fsroot"

TAG="${TAG:-rlv5_4b}"
STEPS="${STEPS:-20}"
SEGMENTS="${SEGMENTS:-3}"
SAVE_FREQ="${SAVE_FREQ:-5}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
# WALLTIME is a knob because of maintenance reservations. sbatch will not start a job
# that would still be running when a MAINT reservation opens, so during the run-up to
# one a 23:59:00 request sits at "ReqNodeNotAvail, Reserved for maintenance" and burns
# the whole window idle. Check `scontrol show reservation` and pass a WALLTIME that
# ends before StartTime; the chained segments resume from the checkpoint afterwards, so
# a short first segment costs nothing but the restart.
WALLTIME="${WALLTIME:-23:59:00}"

RL_PATH="$D/envs/rl/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin"

COMMON="TOTAL_TRAINING_STEPS=$STEPS,TRAIN_BATCH_SIZE=64,PPO_MINI_BATCH_SIZE=16,ROLLOUT_N=16,SAVE_FREQ=$SAVE_FREQ,\
MAX_ACTOR_CKPT_TO_KEEP=$MAX_ACTOR_CKPT_TO_KEEP,\
ADAPTIVE_N_ENABLE=0,ADAPTIVE_N_INWAVE=1,ADAPTIVE_N_OVERLAP=0,ADAPTIVE_N_MAX=32,ADAPTIVE_N_MAX_EXTRA=512,\
ADAPTIVE_N_MAX_EXTRA_PER_WORKER=256,ADAPTIVE_N_AGENTS=single_turn_agent,\
FS_OVERLONG_PENALTY=0,\
FS_OVERLONG_FILTER=0,LOSS_AGG_MODE=seq-mean-token-mean,\
MAX_NUM_SEQS=128,GPU_MEMORY_UTILIZATION=0.90,ACTOR_PARAM_OFFLOAD=True,ACTOR_OPTIMIZER_OFFLOAD=True,\
FRONTIERSMITH_SYNTH_MAX_CONC=4,FSX_CHILD_MEM_MB=8192,MLS_RL_EPISODE_MEM_MB=16384,\
MLS_RL_MAX_TESTS=1,MLS_RL_EPISODE_TIMEOUT=3000,\
MAX_PROMPT_LENGTH=26624,MAX_MODEL_LEN=59392,\
TMPDIR=$D/tmp,TMP=$D/tmp,TEMP=$D/tmp,\
TRAIN_DATA=$D/fsroot/data/multisource_rl/train_time2026.parquet,\
PATH=$RL_PATH,\
MLS_RL_MLSBENCH_ROOT=$D/mlsroot_train,\
MLS_RL_WORKER_PYTHON=$D/envs/client/bin/python3,\
JULIA_DEPOT_PATH=$D/envs/research_julia/julia_depot,\
PYTHON_JULIAPKG_PROJECT=$D/envs/research_julia/julia_env"

declare -A M=(
  [base]=$D/models/Qwen3.5-4B
  [ft01mix_a10]=$D/models/4b_full_wd01_withag_soup10
  [lo32nm_a10]=$D/models/4b_lora_r32_nomaint_soup10
  [ft03nm_a20]=$D/models/4b_full_wd03_nomaint_soup20
)

for arm in ${ARMS_ONLY:-base ft01mix_a10 lo32nm_a10}; do
  [ -f "${M[$arm]}/config.json" ] || { echo "FATAL: no config.json in ${M[$arm]}" >&2; exit 1; }
  prev=""
  for seg in $(seq 1 "$SEGMENTS"); do
    name="rl4b-$arm"; [ "$seg" -gt 1 ] && name="rl4b-$arm-c$seg"
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    j=$(sbatch --parsable --job-name="$name" --mem=650G --time="$WALLTIME" $dep \
      --account=chij --qos=short \
      --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.err" \
      --export=ALL,EXPERIMENT_NAME=${TAG}_$arm,MODEL_PATH=${M[$arm]},$COMMON \
      slurm/cc_rl_multisource.sh)
    prev=$j
    echo -n "$j "
  done
  echo "<- rl4b-$arm  (${M[$arm]##*/})"
done

#!/usr/bin/env bash
# Submit GRPO RL on frontiersmith_synth (500 synthetic problems) -- the
# "RL on our synthetic problems" arm, to compare against the FrontierCS arm
# (scripts/rlm_amplify_v3_submit.sh). Same GRPO SPEED config (rlm_amplify_v3),
# but TRAIN_DATA = the synth parquet and the reward is the offline synth harness
# (no judge server). Mirrors rlm_amplify_v3_submit.sh model set + GPU scaling.
#
# Prereq: python scripts/prepare_frontiersmith_synth_parquet.py  (builds the parquet;
#         the launcher also self-builds it if missing).
#
# Usage:  bash scripts/cc_rl_frontiersmith_synth_submit.sh              # default set, 4 GPU
#         GPUS=2 bash scripts/cc_rl_frontiersmith_synth_submit.sh       # 2 GPU
#         ONLY="q35_inst_start q35_sft" bash scripts/cc_rl_frontiersmith_synth_submit.sh
set -euo pipefail

ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
SCRIPT="$FS/slurm/cc_rl_frontiersmith_synth.sh"
TRAIN="$FS/data/frontiersmith_synth/train.parquet"
VAL="$FS/data/frontiersmith_synth/full.parquet"
SYNTH_ROOT="$ROOT/innovation_prior/frontiersmith_synth"
GPUS="${GPUS:-4}"
STEPS="${STEPS:-40}"
SAVE="${SAVE:-5}"
CPUS=$(( GPUS * 8 ))     # ailab cap: 8 cores/GPU
MEMG=$(( GPUS * 120 ))
# Length config = the PROVEN winner run rlfsx_q35_inst_start (job 10728651, 4 GPU,
# 20 steps in 8h41, final FCS 7.61 > base): response 32768 / model_len 45056.
# The launcher's 20k default truncates synth responses (~87% clip on SFT models)
# and starves the reward. 2-GPU at 32k OOMs in update_actor (5/5 failures
# 10979699-703) -- use GPUS=4 for 32k.
MAXRESP="${MAXRESP:-32768}"
MAXLEN="${MAXLEN:-45056}"
# Batch shape (2026-07-11 user directive): 32 queries x 8 rollouts = 256 seqs/step
# (winner ran 8x4=32). MB is in PROMPT units (verl multiplies by rollout.n):
# MB=16 -> 16x8 = 128 sequences per gradient update -> 2 updates per iteration
# (user: 64-128 seqs/update, <=4 updates/iter). KL + loss stay verl GRPO defaults
# (use_kl_loss=True coef 0.001 low_var_kl, adv=grpo std-normalized).
# ~8.4M tokens/step => expect ~1.5-2h/step on 4 GPU early (shrinks as RL shortens
# responses); 20 steps will NOT fit one 23:59 window, so each run auto-chains a
# continuation job (FRESH_START=0 -> verl resume_mode=auto picks up the last save).
TB="${TB:-32}"
MB="${MB:-16}"
RN="${RN:-8}"
GMU="${GMU:-0.50}"
# Rollout sampling matched to the EVAL protocol (2026-07-11 user directive:
# "rollout 的参数要和 evaluation 保持一致"): temp 1.0 (already), top_p 0.95,
# top_k 20, presence_penalty 1.5, response 32768. presence_penalty reaches the
# agent loop via env (single-node Ray inherits the driver env).
RTOPP="${RTOPP:-0.95}"
RTOPK="${RTOPK:-20}"
RPP="${RPP:-1.5}"
# Keep ALL periodic saves (verl's default pruning deleted step5/10 before we could
# merge+eval them in the mixed arm -- never again).
KEEP="${KEEP:-10}"

# Build the parquet up front if it's not there (the launcher can also self-build).
if [ ! -f "$TRAIN" ]; then
  echo "[submit] $TRAIN missing -> building it."
  ( cd "$FS" && { [ -f .venv-vllm023/bin/activate ] && source .venv-vllm023/bin/activate; } ; \
    python scripts/prepare_frontiersmith_synth_parquet.py --synth-root "$SYNTH_ROOT" )
fi

declare -A MODELS=(
  # 2026-08-03 wave: RL restart on the new-matrix SFT winners (user program)
  [lora_im_s01]="$ROOT/models_sft/lora_q35_im_r32_s01_merged"
  [soup_im_all_a20]="$ROOT/models_sft/soup_q35_im_all_ft_a20"
  [soup_innnew_a10]="$ROOT/models_sft/soup_q35_innnew_ft_a10"
  [q3_inst_start]="$FS/models/Qwen3-8B"
  [q35_inst_start]="$FS/models/Qwen3.5-9B-bf16"
  [q35_base_start]="$FS/models/rlm_Qwen3.5-9B-Base-chat"
  [q35_sft]="$ROOT/models_sft/sft_q35_a100_method"
  [q3_sft]="$ROOT/models_sft/sft_q3_a100_method"
  [q35_soup10]="$ROOT/models_sft/soup_q35_a100_method_soupa10"
  [q3_soup10]="$ROOT/models_sft/soup_q3_a100_method_soupa10"
  [q35_base_soup20]="$ROOT/models_sft/soup_q35_a00_method_soupa20"
  # clean round (decontam+traj) soups -- the 2026-07 synth-RL arm
  [cl_nom_a5]="$ROOT/models_sft/soup_q35_clean_nomaintain_wd01_a5"
  [cl_nom_a10]="$ROOT/models_sft/soup_q35_clean_nomaintain_wd01_a10"
  [cl_nom_a20]="$ROOT/models_sft/soup_q35_clean_nomaintain_wd01_a20"
  [cl_newmt_a10]="$ROOT/models_sft/soup_q35_clean_full_newmt_a10"
  [cl_wd03_a10]="$ROOT/models_sft/soup_q35_clean_nom_wd03_a10"
)
# Default arm: start controls + good sft/soup (same shortlist as the v3 expansion).
ORDER=(q35_inst_start q35_sft q35_soup10 q3_inst_start q3_sft q3_soup10 q35_base_start q35_base_soup20)
[ -n "${ONLY:-}" ] && ORDER=($ONLY)

cd "$FS"
for tag in "${ORDER[@]}"; do
  mp="${MODELS[$tag]}"
  if [ ! -f "$mp/config.json" ]; then
    echo "ABORT: model missing for $tag: $mp" >&2; exit 1
  fi
  arch=$(python3 -c "import json;print(json.load(open('$mp/config.json'))['architectures'][0])" 2>/dev/null || echo "?")
  if [[ "$arch" == *ForConditionalGeneration* ]]; then
    for f in preprocessor_config.json processor_config.json video_preprocessor_config.json chat_template.jinja; do
      [ -f "$mp/$f" ] || cp -n "$FS/models/Qwen3.5-9B-bf16/$f" "$mp/$f" 2>/dev/null || true
    done
  fi
  exp="fsx_$tag"
  ck="$FS/checkpoints/rl_frontiersmith_synth/$exp"
  ro="$FS/outputs/rl_frontiersmith_synth_rollout/$exp"
  COMMON="MODEL_PATH=$mp,TRAIN_DATA=$TRAIN,VAL_DATA=$VAL,EXPERIMENT_NAME=$exp,PROJECT_NAME=rl_frontiersmith_synth,CKPT_DIR=$ck,ROLLOUT_DIR=$ro,FRONTIERSMITH_SYNTH_ROOT=$SYNTH_ROOT,FRONTIERSMITH_SYNTH_FAIL_SOFT=1,TOTAL_TRAINING_STEPS=$STEPS,SAVE_FREQ=$SAVE,TEST_FREQ=100000,NGPU=$GPUS,MAX_RESPONSE_LENGTH=$MAXRESP,MAX_MODEL_LEN=$MAXLEN,MAX_NUM_BATCHED_TOKENS=$MAXLEN,TRAIN_BATCH_SIZE=$TB,PPO_MINI_BATCH_SIZE=$MB,ROLLOUT_N=$RN,MAX_ACTOR_CKPT_TO_KEEP=$KEEP,GPU_MEMORY_UTILIZATION=$GMU,ROLLOUT_TOP_P=$RTOPP,ROLLOUT_TOP_K=$RTOPK,ROLLOUT_PRESENCE_PENALTY=$RPP"
  jid=$(sbatch --parsable \
    --job-name="rlfsx_$tag" \
    --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="${MEMG}G" \
    --export=ALL,$COMMON,FRESH_START=1 \
    "$SCRIPT")
  # Continuation windows: start when the previous window ends (walltime or done) and
  # resume from the last saved step. The launcher's window-guard makes them true
  # no-ops once TOTAL_TRAINING_STEPS is reached.
  w2=$(sbatch --parsable --dependency=afterany:$jid \
    --job-name="rlfsx_${tag}_w2" \
    --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="${MEMG}G" \
    --export=ALL,$COMMON,FRESH_START=0 \
    "$SCRIPT")
  w3=$(sbatch --parsable --dependency=afterany:$w2 \
    --job-name="rlfsx_${tag}_w3" \
    --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="${MEMG}G" \
    --export=ALL,$COMMON,FRESH_START=0 \
    "$SCRIPT")
  echo "submitted rlfsx_$tag -> $jid (w2=$w2 w3=$w3)  (model=$mp, ${GPUS}GPU, ${TB}x${RN} mini=$MB, arch=$arch)"
done

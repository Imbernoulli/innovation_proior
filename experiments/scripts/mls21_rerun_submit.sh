#!/usr/bin/env bash
# mls21_rerun_submit.sh -- re-run MLS-21 with the two infrastructure ceilings lifted.
# User 2026-09-13: "要20多道题都测完啊，优先01mix测完" + "分母必须一样啊，都是21题".
#
# Why a rerun is needed at all (all verified in the task logs, not guessed):
#   1. MAX_MODEL_LEN defaulted to 40960 while Qwen3.5 supports 262144, so multi-turn
#      agents died with openai.BadRequestError "maximum context length is 40960".
#      That is what produced most `agent_failed+scored` cells.
#   2. MLSBENCH_PY defaulted to bl3615's python, which has neither `causallearn`
#      nor `deap`; the driver imports holdout/<task>/dgp.py, so those two tasks
#      (causal-observational-linear-gaussian, optimization-multi-objective) died
#      before the model was ever queried. $D/envs/client has both -- verified by
#      importing both dgp.py files successfully.
#   3. TASK_TIMEOUT 7200 starved optimization-evolution-strategy.
# Every arm must be rerun under the SAME new config or the denominator argument breaks.
#
#   bash scripts/mls21_rerun_submit.sh <ARM> [<ARM> ...]
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
declare -A MP=(
  [base9b_v2c]="$FS/models/Qwen3.5-9B-bf16"
  [ft01mix_a10]="$D/models/full_wd01_withag_v2b_soup10"
  [ft03nm_a20]="$D/models/full_wd03_nomaint_soup20"
  [lo32nm_a10]="$D/models/lora_r32_nomaint_soup10"
  [rlv5_base_s20]="$D/models/rlv5_base_s20"
  [rlv5_ft01mix_a10_s20]="$D/models/rlv5_ft01mix_a10_s20"
  [rlv5_ft03nm_a20_s20]="$D/models/rlv5_ft03nm_a20_s20"
  [rlv5_lo32nm_a10_s20]="$D/models/rlv5_lo32nm_a10_s20"
  [base4b]="$D/models/Qwen3.5-4B"
  [4b_ft01mix_a10]="$D/models/4b_full_wd01_withag_soup10"
  [4b_lo32nm_a10]="$D/models/4b_lora_r32_nomaint_soup10"
  [rlv5_4b_base_s20]="$D/models/rlv5_4b_base_s20"
  [rlv5_4b_ft01mix_a10_s20]="$D/models/rlv5_4b_ft01mix_a10_s20"
  [rlv5_4b_lo32nm_a10_s20]="$D/models/rlv5_4b_lo32nm_a10_s20"
)
TASKS21="causal-discovery-discrete causal-observational-linear-gaussian causal-observational-linear-non-gaussian causal-observational-nonlinear causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy optimization-hyperparameter-search optimization-multi-objective optimization-nas optimization-online-bandit"
LOG=$D/logs/mls21_rerun_jobids.txt
mkdir -p $D/logs/locks
SUF="${SUF:-r2}"          # new TAG suffix so the old runs are never overwritten
MML="${MML:-98304}"       # was 40960
TTO="${TTO:-10800}"       # was 7200
CONC="${CONC:-5}"         # was 7; longer contexts need more KV per sequence
WALL="${WALL:-20:00:00}"  # was 12h; must stay < 24h

dual() {
  local name="$1" t="$2" env="$3" script="$4" a g
  local lock="$D/logs/locks/${name}.lock"
  [ -e "$lock" ] && { echo "SKIP (lock exists): $name"; return; }
  : > "$lock"
  a=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G --time="$t" \
      -J "$name" -o "$D/logs/${name}-%j.out" \
      "--export=ALL,${env},DUAL_NAME=${name},REAL_SCRIPT=${script}" "$D/slurm_overlay/cc_dual_wrap.sh") || { echo "sbatch ailab failed: $name"; return; }
  g=$(sbatch --parsable --account=chij --qos=gpu-short --constraint=gpu80 --gres=gpu:1 -c 8 --mem=200G --time="$t" \
      -J "$name" -o "$D/logs/${name}-%j.out" \
      "--export=ALL,${env},DUAL_NAME=${name},REAL_SCRIPT=${script}" "$D/slurm_overlay/cc_dual_wrap.sh") || g=none
  echo "$name ailab=$a gpu=$g" | tee -a "$LOG"
}

for ARM in "$@"; do
  M="${MP[$ARM]:?unknown arm $ARM}"; [ -e "$M/config.json" ] || { echo "no model $M"; continue; }
  TAG="${ARM}_${SUF}"
  cd "$FS"   # mlsbench script takes PROJECT_ROOT from SLURM_SUBMIT_DIR
  dual "mls21-$TAG" "$WALL" "MODEL_PATH=$M,TAG=$TAG,OUTPUT_BASE=$D/outputs/cc_mls21_$TAG,MLSBENCH_ROOT=$D/mlsroot,MLSBENCH_DATA_ROOT=$D/mlsvendor/data,MLSBENCH_PY=$D/envs/client/bin/python,MAX_MODEL_LEN=$MML,TASK_TIMEOUT=$TTO,HF_HOME=$D/.hf,VLLM_VENV=$D/envs/vllm023,VLLM_CACHE_DIR=$D/.cache/vllm,CONCURRENCY=$CONC,VLLM_PORT=$((41000 + RANDOM % 20000)),TASKS=$TASKS21" "$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh"
done

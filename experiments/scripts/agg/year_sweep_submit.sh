#!/usr/bin/env bash
# year_sweep_submit.sh -- test-time year sweep (user request 2026-09-11).
# Same models, same benches, same sampling protocol as the 2026 numbers; the ONLY change is
# the year in the system prompt: FCS/ALE/research use "It is now year {Y}. You are a good
# researcher." (EVAL_SYS_PROMPT_MODE=short, exactly what the 2026 runs printed), MLS uses
# MLSBENCH_SYS_PREFIX "It is now year {Y}." Outputs land under TAG=<arm>_y<Y> so allstats.py
# / mls_merge.py aggregate them as ordinary arms.
#   bash scripts/year_sweep_submit.sh <YEAR> <ARM> [<ARM> ...]
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
YEAR="${1:?year}"; shift
declare -A MP=(
  [base9b_v2c]="$FS/models/Qwen3.5-9B-bf16"
  [lo32nm_a10]="$D/models/lora_r32_nomaint_soup10"
  [rlv5_lo32nm_a10_s20]="$D/models/rlv5_lo32nm_a10_s20"
  [base4b]="$D/models/Qwen3.5-4B"
  [4b_lo32nm_a10]="$D/models/4b_lora_r32_nomaint_soup10"
  [rlv5_4b_lo32nm_a10_s20]="$D/models/rlv5_4b_lo32nm_a10_s20"
)
TASKS21="causal-discovery-discrete causal-observational-linear-gaussian causal-observational-linear-non-gaussian causal-observational-nonlinear causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy optimization-hyperparameter-search optimization-multi-objective optimization-nas optimization-online-bandit"
LOG=$D/logs/year_sweep_jobids.txt
mkdir -p $D/logs/locks
dual() {  # name walltime env-string real-script
  local name=$1 t=$2 env=$3 script=$4 a g
  [ -d "$D/logs/locks/$name.lock" ] && { echo "SKIP $name: lock exists (already ran)"; return; }
  a=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G --time="$t" \
      --job-name="$name" --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.out" \
      "--export=ALL,${env},DUAL_NAME=${name},REAL_SCRIPT=${script}" "$D/slurm_overlay/cc_dual_wrap.sh") || { echo "sbatch ailab failed: $name"; return; }
  g=$(sbatch --parsable --account=chij --qos=gpu-short --constraint=gpu80 --gres=gpu:1 -c 8 --mem=200G --time="$t" \
      --job-name="$name" --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.out" \
      "--export=ALL,${env},DUAL_NAME=${name},REAL_SCRIPT=${script}" "$D/slurm_overlay/cc_dual_wrap.sh") || g=none
  echo "$a $g" > "$D/logs/locks/$name.ids"
  echo "$name ailab=$a gpu=$g" | tee -a "$LOG"
}
for ARM in "$@"; do
  M="${MP[$ARM]:?unknown arm $ARM}"; [ -e "$M/config.json" ] || { echo "no model $M"; continue; }
  TAG="${ARM}_y${YEAR}"
  cd "$D/fsroot"
  for s in 0 1; do
    dual "ev-$TAG-f$s" 08:00:00 "MODEL=$M,TAG=$TAG,SOURCE=both,NUM_SHARDS=2,SHARD_IDX=$s,EVAL_RESEARCHER_YEAR=$YEAR" "$D/slurm_overlay/cc_eval_allinone_ailab.sh"
    dual "ev-$TAG-r$s" 04:00:00 "MODEL=$M,TAG=$TAG,SOURCE=research,NUM_SHARDS=2,SHARD_IDX=$s,EVAL_RESEARCHER_YEAR=$YEAR" "$D/slurm_overlay/cc_eval_allinone_ailab.sh"
  done
  cd "$FS"   # mlsbench script takes PROJECT_ROOT from SLURM_SUBMIT_DIR
  dual "mls21-$TAG" 12:00:00 "MODEL_PATH=$M,TAG=$TAG,OUTPUT_BASE=$D/outputs/cc_mls21_$TAG,MLSBENCH_ROOT=$D/mlsroot,MLSBENCH_DATA_ROOT=$D/mlsvendor/data,HF_HOME=$D/.hf,VLLM_VENV=$D/envs/vllm023,VLLM_CACHE_DIR=$D/.cache/vllm,EVAL_RESEARCHER_YEAR=$YEAR,MLSBENCH_SYS_PREFIX=It is now year $YEAR.,CONCURRENCY=7,TASK_TIMEOUT=7200,VLLM_PORT=$((41000 + RANDOM % 20000)),TASKS=$TASKS21" "$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh"
done

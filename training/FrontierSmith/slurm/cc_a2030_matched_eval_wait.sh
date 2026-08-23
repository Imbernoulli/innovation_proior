#!/usr/bin/env bash
# Wait for a20/a30 RL-after HF exports, then run MATCHED (32k) FCS + MLS-devfix on each.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith
START=$FS/models/Qwen3.5-9B-bf16
for i in $(seq 1 240); do
  r20=$ROOT/models/rl_soup_mtv4_a20_step40_hf/config.json
  r30=$ROOT/models/rl_soup_mtv4_a30_step40_hf/config.json
  { [ -e "$r20" ] && ls $ROOT/models/rl_soup_mtv4_a20_step40_hf/*.safetensors >/dev/null 2>&1 \
    && [ -e "$r30" ] && ls $ROOT/models/rl_soup_mtv4_a30_step40_hf/*.safetensors >/dev/null 2>&1; } && break
  sleep 180
done
echo "a20/a30 HF exports ready; launching matched FCS + MLS"
cd "$FS"
for tag in rl_soup_mtv4_a20 rl_soup_mtv4_a30; do
  d=$ROOT/models/${tag}_step40_hf
  cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json "$d/" 2>/dev/null
  sbatch --parsable --job-name="cc-eval-rlafter_${tag}" --time=06:00:00 \
    --export=ALL,MODEL_PATH=$d,TAG=rlafter_${tag} slurm/cc_eval_thinking_both_ailab.sh | xargs echo "  FCS rlafter_$tag ->"
  sbatch --parsable --job-name="cc-mlsdevfix-rlafter_${tag}" --time=08:00:00 \
    --export=ALL,MODEL_PATH=$d,TAG=rlafter_${tag}_devfix,MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev,EVAL_RESEARCHER_YEAR=2026 \
    slurm/cc_eval_mlsbench_cpu_ailab.sh | xargs echo "  MLS rlafter_$tag ->"
done

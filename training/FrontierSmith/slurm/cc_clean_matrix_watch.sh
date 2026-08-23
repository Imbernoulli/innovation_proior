#!/usr/bin/env bash
# clean matrix: when clean SFT lands -> soup(a5/10/20/30/50, WORKING cmd) -> eval FCS+ALE / MLS / Research.
# Fixes the Codex soup bug (source envs/sft_lf/bin/activate -> nonexistent). Singleton+idempotent.
set -uo pipefail
exec 205>/tmp/cc_clean_matrix.lock; flock -n 205 || exit 0
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith
ENV=$ROOT/envs/sft_lf; START=$FS/models/Qwen3.5-9B-bf16; SOUP=$FS/scripts/cc_model_soup_merge.py
ELS=$FS/slurm/cc_eval_thinking_both_ailab.sh; MLS=$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh; RES=$FS/slurm/cc_eval_research_ailab.sh
PROC="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"
have(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
qd(){ squeue -u $USER -h -n "$1" -o %T 2>/dev/null|grep -q .; }
ev(){ local mp=$1 t=$2; # FCS+ALE, MLS, Research
  { [ -d "$FS/outputs/cc_eval_${t}_thinking_32k_both_vllm" ]||qd cc-eval-$t; }||sbatch -J cc-eval-$t --time=06:00:00 --export=ALL,MODEL_PATH=$mp,TAG=$t,CONCURRENCY=16,FRONTIERCS_STRIP_THINK_EXTRACT=1 $ELS >/dev/null 2>&1&&echo "  eval-FCS/ALE $t"
  { ls $FS/outputs/cc_mlsbench_cpu_${t}/summary.json >/dev/null 2>&1||qd cc-mls-$t; }||sbatch -J cc-mls-$t --time=08:00:00 --export=ALL,MODEL_PATH=$mp,TAG=$t,MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev,EVAL_RESEARCHER_YEAR=2026 $MLS >/dev/null 2>&1&&echo "  eval-MLS $t"
  { [ -d "$FS/outputs/cc_eval_${t}_research_thinking_32k_vllm" ]||qd cc-res-$t; }||sbatch -J cc-res-$t --time=08:00:00 --export=ALL,MODEL_PATH=$mp,TAG=$t,RESEARCH_SCOPE=full $RES >/dev/null 2>&1&&echo "  eval-RES $t"; }
for i in $(seq 1 288); do
  for m in clean_full_wd01 clean_nomaintain_wd01; do
    S=$ROOT/models_sft/sft_q35_$m; have "$S"||continue
    $PROC "$S"/ 2>/dev/null
    ev "$S" clean_${m}_sft
    for a in 5 10 20 30 50; do
      SP=$ROOT/models_sft/soup_q35_${m}_a${a}
      if ! have "$SP" && ! qd soup2_${m}_a$a; then
        af=$(python3 -c "print($a/100)")
        sbatch -J soup2_${m}_a$a --partition=cpu --cpus-per-task=8 --mem=200G --time=01:30:00 -o $FS/logs/%x-%j.out \
          --wrap "$ENV/bin/python $SOUP --sft $S --base $START --alpha $af --out $SP && $PROC $SP/" >/dev/null 2>&1 && echo "  soup $m a$a"
      fi
      have "$SP" && ev "$SP" clean_${m}_a${a}
    done
  done
  sleep 300
done

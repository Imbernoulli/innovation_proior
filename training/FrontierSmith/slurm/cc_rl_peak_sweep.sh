#!/usr/bin/env bash
# Export + clean-eval the RL step-sweep to find each model's PEAK (FrontierSmith peaks ~step10).
# merge verl->HF (CPU) --afterok--> cc_eval (GPU, concurrency 16, best@5+mean@5). Idempotent.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; START=$FS/models/Qwen3.5-9B-bf16
VENV=$FS/.venv-vllm023; MERGE=$FS/scripts/merge_fsdp_to_hf.py
PROC="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"
for tag in rlpeak_start rlpeak_methodv4_r2_a30; do
  for s in 5 10 15 20; do
    ck=$FS/checkpoints/rlm_amplify_v3/$tag/global_step_$s; hf=$ROOT/models/${tag}_s${s}_hf; T=peak_${tag}_s$s
    [ -d "$ck/actor" ] || { echo "  [skip] no ckpt $tag s$s"; continue; }
    [ -d "$FS/outputs/cc_eval_${T}_thinking_32k_both_vllm" ] && { echo "  [done] $T"; continue; }
    if ls "$hf"/model*.safetensors >/dev/null 2>&1; then dep="";
    else
      jm=$(sbatch --parsable -J mrg-$tag-s$s --partition=cpu --cpus-per-task=8 --mem=220G --time=01:30:00 \
        -o $FS/logs/%x-%j.out -e $FS/logs/%x-%j.err \
        --wrap "source $VENV/bin/activate && python $MERGE --ckpt $ck --output $hf --dtype bfloat16 && $PROC $hf/")
      dep="--dependency=afterok:$jm"
    fi
    sbatch -J cc-eval-$T $dep --time=06:00:00 --export=ALL,MODEL_PATH=$hf,TAG=$T,CONCURRENCY=16,FRONTIERCS_JUDGE_MAX_WAIT=1200 \
      $FS/slurm/cc_eval_thinking_both_ailab.sh >/dev/null 2>&1 && echo "  [submit] $T ${dep:+(merge $jm)}"
  done
done

#!/usr/bin/env bash
# Wait until a20 & a30 RL reach global_step_40, then export verl->HF + matched FCS/ALE eval for each.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith
for i in $(seq 1 360); do
  d20=$(ls -d $FS/checkpoints/rlm_amplify_v3/r1_soup_methodtraj_v4_a20/global_step_40 2>/dev/null)
  d30=$(ls -d $FS/checkpoints/rlm_amplify_v3/r1_soup_methodtraj_v4_a30/global_step_40 2>/dev/null)
  [ -n "$d20" ] && [ -n "$d30" ] && break
  sleep 180
done
echo "a20/a30 RL hit step 40; launching export+matched-eval"
i=10
for tag in rl_soup_mtv4_a20 rl_soup_mtv4_a30; do
  src=r1_soup_methodtraj_v4_${tag##*_}   # a20 / a30
  ckpt="$FS/checkpoints/rlm_amplify_v3/$src/global_step_40"
  vp=$((8000+i*120)); fp=$((8082+i*120)); gj=$((5050+i*30)); i=$((i+1))
  # 1) export verl->HF (+ its own eval, ignored) -- reliable export path
  sbatch --job-name="cc-rleval-$tag" --gres=gpu:2 --cpus-per-task=16 --mem=400G --time=12:00:00 \
    --export=ALL,CKPT_PATH="$ckpt",MODEL_OUTPUT_DIR="$ROOT/models/${tag}_step40_hf",OUTPUT_DIR="$FS/outputs/eval_${tag}_step40_vllm",SERVED_MODEL_NAME="$tag",VLLM_PORT=$vp,PORT=$fp,GJ_PORT=$gj \
    "$FS/slurm/export_and_eval_qwen35_ckpt_vllm_ailab.sh"
done
echo "exports launched; matched cc_eval_thinking on the HF models must be run once models/<tag>_step40_hf exists"

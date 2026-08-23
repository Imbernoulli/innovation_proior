#!/usr/bin/env bash
# Fire cc_eval_all_benchmarks (FCS+ALE+MLS+TTT+Theta) on every r3 soup (model x alpha) when it lands.
# Singleton (flock) + squeue-dedup + idempotent (skip if summary_all.json exists). No runaway.
set -uo pipefail
exec 201>/tmp/cc_all_eval_watch.lock; flock -n 201 || exit 0
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; EB=$FS/slurm/cc_eval_all_benchmarks.sh
MODELS=(methodv4_r3 methodtraj_v4_r3 full_r3_wd0 full_r3_wd01)
ALPHAS=(10 20 30 50)
have(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
for i in $(seq 1 288); do
  for m in "${MODELS[@]}"; do for a in "${ALPHAS[@]}"; do
    soup=$ROOT/models_sft/soup_q35_${m}_a${a}; tag=r3_${m}_a${a}
    have "$soup" || continue
    [ -e "$FS/outputs/cc_eval_all_${tag}/summary_all.json" ] && continue
    squeue -u $USER -h -n cc-all-$tag -o %T 2>/dev/null | grep -q . && continue
    sbatch -J cc-all-$tag --gres=gpu:1 --cpus-per-task=8 --mem=240G --time=12:00:00 \
      --export=ALL,MODEL_PATH=$soup,TAG=$tag,FRONTIERCS_STRIP_THINK_EXTRACT=1 $EB >/dev/null 2>&1 \
      && echo "$(date +%H:%M) fired all-benchmark $tag"
  done; done
  sleep 600
done

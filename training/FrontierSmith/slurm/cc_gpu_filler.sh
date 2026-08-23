#!/usr/bin/env bash
# Robust GPU saturator: keep ailab GPUs busy with USEFUL variance re-runs of the key
# evals (research is high-variance -> repeats give error bars). Singleton (flock).
# Whenever my running GPU jobs < TARGET, submit the next un-done (model, eval, version)
# from the backlog. Idempotent by output-dir. Bounded to VMAX versions => won't run forever.
set -uo pipefail
exec 202>/tmp/cc_gpu_filler.lock; flock -n 202 || exit 0
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith
RES=$FS/slurm/cc_eval_research_ailab.sh; ELS=$FS/slurm/cc_eval_thinking_both_ailab.sh
START=$FS/models/Qwen3.5-9B-bf16
TARGET=${TARGET:-15}; VMAX=${VMAX:-3}
# key models to get variance error-bars on (research = noisy, most valuable)
declare -A M=( [start]=$START )
for m in methodv4_r3 methodtraj_v4_r3 full_r3_wd0 full_r3_wd01; do for a in 10 20 30 50; do
  d=$ROOT/models_sft/soup_q35_${m}_a${a}; [ -e "$d/config.json" ] && M[${m}_a${a}]=$d
done; done
gpu(){ squeue -u $USER -h -t RUNNING -o "%b" 2>/dev/null|grep -oE 'gpu:[0-9]+'|awk -F: '{s+=$2}END{print s+0}'; }
qd(){ squeue -u $USER -h -o "%j" 2>/dev/null|grep -qx "$1"; }
for i in $(seq 1 576); do
  for v in $(seq 2 $VMAX); do for k in "${!M[@]}"; do
    [ "$(gpu)" -ge "$TARGET" ] && break 2
    mp=${M[$k]}; t=var_${k}_v$v
    # research variance re-run (idempotent)
    if [ ! -d "$FS/outputs/cc_eval_${t}_research_thinking_32k_vllm" ] && ! qd "cc-res-$t"; then
      sbatch -J cc-res-$t --time=08:00:00 --export=ALL,MODEL_PATH=$mp,TAG=$t,RESEARCH_SCOPE=full $RES >/dev/null 2>&1 \
        && echo "$(date +%H:%M) fill research $t (gpu=$(gpu))"
    fi
  done; done
  sleep 300
done

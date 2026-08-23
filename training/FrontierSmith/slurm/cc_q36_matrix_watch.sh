#!/usr/bin/env bash
# Wait for each Qwen3.6 FP8 download to complete, copy proc configs if missing,
# then submit the FCS eval MATRIX per model (strip口径): {32k, 48k} budgets.
# 2 models x 2 budgets = the "2 matrices". Idempotent.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; ELS=$FS/slurm/cc_eval_thinking_both_ailab.sh
SRC=$FS/models/Qwen3.5-9B-bf16  # proc-config donor (same qwen3_5 family)
declare -A M=( [q36_27b]=$ROOT/models/Qwen3.6-27B-FP8 [q36_35bA3b]=$ROOT/models/Qwen3.6-35B-A3B-FP8 )
done_tag(){ [ -d "$FS/outputs/cc_eval_$1_thinking_32k_both_vllm" ]; }
sub(){ sbatch -J cc-eval-$1 --time=10:00:00 --export=ALL,MODEL_PATH=$2,TAG=$1,CONCURRENCY=16,FRONTIERCS_STRIP_THINK_EXTRACT=1,TEMPERATURE=0.6,TOP_P=0.95,TOP_K=20,PRESENCE_PENALTY=0,MAX_TOKENS=$3 $ELS >/dev/null 2>&1 && echo "  submit $1 (budget $3)"; }
for i in $(seq 1 120); do
  allsub=1
  for name in "${!M[@]}"; do
    d=${M[$name]}
    if ls "$d"/*.safetensors >/dev/null 2>&1 && [ -e "$d/config.json" ] && ! pgrep -f "download.*$(basename $d)" >/dev/null 2>&1; then
      for f in preprocessor_config.json processor_config.json video_preprocessor_config.json; do [ -e "$d/$f" ] || cp -n "$SRC/$f" "$d/" 2>/dev/null; done
      done_tag "big32k_$name"  || sub "big32k_$name" "$d" 32768
      done_tag "big48k_$name"  || sub "big48k_$name" "$d" 49152
    else allsub=0; fi
  done
  [ "$allsub" = 1 ] && { echo "both models submitted"; break; }
  sleep 120
done

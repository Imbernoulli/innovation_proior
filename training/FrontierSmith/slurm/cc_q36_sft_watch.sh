#!/usr/bin/env bash
# Wait for each Qwen3.6 bf16 download to complete, then launch LoRA SFT (methodv4_r2
# innovation data) on it. Runs in parallel with the baseline eval. Idempotent.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; LF=$ROOT/LF-innov; TPL=$LF/examples/train_lora/auto/lora_q35_a100_method_r32.yaml
declare -A M=( [q36_27b]=$ROOT/models/Qwen3.6-27B [q36_35bA3b]=$ROOT/models/Qwen3.6-35B-A3B )
complete(){ # dir: index present, all shards present, no active download
  local d=$1; local idx=$d/model.safetensors.index.json
  [ -e "$d/config.json" ] || return 1
  ls $d/*.safetensors.index.json >/dev/null 2>&1 || return 1
  pgrep -f "download Qwen/Qwen3.6.*$(basename $d)$" >/dev/null 2>&1 && return 1
  # shard count vs index
  local want=$(python3 -c "import json,glob;f=glob.glob('$d/*.index.json')[0];print(len(set(json.load(open(f))['weight_map'].values())))" 2>/dev/null)
  local have=$(ls $d/*.safetensors 2>/dev/null|wc -l)
  [ -n "$want" ] && [ "$have" -ge "$want" ]
}
for i in $(seq 1 240); do
  allgo=1
  for name in "${!M[@]}"; do
    d=${M[$name]}; out=$ROOT/models_sft/lora_${name}_methodv4_r2; cfg=$LF/examples/train_lora/auto/lora_${name}_methodv4_r2.yaml
    if [ -e "$out/adapter_model.safetensors" ] || squeue -u $USER -h -n sft_lora_$name -o %T | grep -q .; then continue; fi
    if complete "$d"; then
      python3 - "$TPL" "$cfg" "$d" "$out" <<'PY'
import sys,re
tpl,cfg,mp,out=sys.argv[1:5]
t=open(tpl).read()
t=re.sub(r'^model_name_or_path: .*$','model_name_or_path: '+mp,t,flags=re.M)
t=re.sub(r'^dataset: .*$','dataset: innovation_methodv4_r2',t,flags=re.M)
t=re.sub(r'^output_dir: .*$','output_dir: '+out,t,flags=re.M)
open(cfg,'w').write(t)
PY
      (cd $LF && sbatch -J sft_lora_$name --gres=gpu:4 --cpus-per-task=32 --mem=480G --time=10:00:00 cc-sft-innov.sh "$cfg") >/dev/null 2>&1 \
        && echo "$(date +%H:%M) launched LoRA SFT $name (methodv4_r2) on $d"
    else allgo=0; fi
  done
  [ "$allgo" = 1 ] && { echo "both SFT launched"; break; }
  sleep 180
done

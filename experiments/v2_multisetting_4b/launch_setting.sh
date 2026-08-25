#!/usr/bin/env bash
# launch_setting.sh <setting: full_wd01|full_wd03|lora_r32> <gpuA,gpuB>
set -euo pipefail
S="$1"; GPUS="$2"
case ",$GPUS," in (*,6,*|*,7,*) [ "${FS_ALLOW_GPU67:-0}" = 1 ] || { echo "set FS_ALLOW_GPU67=1 for gpus 6/7"; exit 1; };; esac
D=/srv/home/bohanlyu/innovation_proior/experiments/v2_multisetting_4b
LF=/srv/home/bohanlyu/LF-innov
mkdir -p /srv/home/bohanlyu/models_sft/v2_multisetting_4b/$S
cd "$LF"
PORT=$((29600 + RANDOM % 100))
nohup env CUDA_VISIBLE_DEVICES="$GPUS" FORCE_TORCHRUN=1 MASTER_PORT=$PORT \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PATH="$LF/.venv/bin:$PATH" VIRTUAL_ENV="$LF/.venv" \
  "$LF/.venv/bin/llamafactory-cli" train "$D/sft_$S.yaml" >> "$D/train_$S.log" 2>&1 &
echo "[$S] PID $! gpus=$GPUS port=$PORT log=$D/train_$S.log"

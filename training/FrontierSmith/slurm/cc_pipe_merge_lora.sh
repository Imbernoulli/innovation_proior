#!/usr/bin/env bash
#SBATCH --job-name=cc-pipe-lora-merge
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
set -uo pipefail
ENV=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf
export PATH="$ENV/bin:$PATH"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cudatoolkit/12.8 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false DISABLE_VERSION_CHECK=1
BASE=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B-bf16
SCRIPT=/scratch/gpfs/CHIJ/bohan/fs/merge_lora_scaled.py
MS=/scratch/gpfs/CHIJ/bohan/fs/models_sft
for arm in $ARMS; do
  ADAPTER=$MS/sft_q35_$arm
  [ -e "$ADAPTER/adapter_config.json" ] || { echo "ERROR: $ADAPTER not a LoRA adapter"; exit 2; }
  for s in $SCALES; do
    OUT=$MS/sft_q35_${arm}_s${s/0./0}_merged
    echo "=== merge $arm scale=$s -> $OUT ==="
    rm -rf "$OUT"
    "$ENV/bin/python" "$SCRIPT" --base "$BASE" --adapter "$ADAPTER" --scale "$s" --out "$OUT" --proc-src "$BASE" 2>&1 | tail -3
    [ -e "$OUT/model.safetensors" ] || [ -e "$OUT/model.safetensors.index.json" ] || { echo "ERROR: no weights"; exit 3; }
  done
done
echo "=== LORA MERGES DONE ==="

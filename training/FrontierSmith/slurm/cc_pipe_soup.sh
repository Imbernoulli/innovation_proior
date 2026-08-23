#!/usr/bin/env bash
#SBATCH --job-name=cc-pipe-soup
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
set -uo pipefail
ENV=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf
export PATH="$ENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BASE=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B-bf16
SCRIPT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/scripts/cc_model_soup_merge.py
MS=/scratch/gpfs/CHIJ/bohan/fs/models_sft
for arm in $ARMS; do
  SFT=$MS/sft_q35_$arm
  [ -e "$SFT/config.json" ] || { echo "ERROR: $SFT missing"; exit 2; }
  for A in $ALPHAS; do
    OUT=$MS/sft_q35_${arm}_soup${A/./}
    echo "=== soup $arm alpha=$A -> $OUT ==="
    rm -rf "$OUT"
    "$ENV/bin/python" "$SCRIPT" --sft "$SFT" --base "$BASE" --alpha "$A" --out "$OUT" 2>&1 | tail -3
    [ -e "$OUT/model.safetensors" ] || [ -e "$OUT/model.safetensors.index.json" ] || { echo "ERROR: no weights"; exit 3; }
  done
done
echo "=== SOUPS DONE ==="

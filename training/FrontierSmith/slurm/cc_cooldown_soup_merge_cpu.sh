#!/usr/bin/env bash
# PHASE 1 cool-down: model-soup merge 0.9*instruct + 0.1*base on a dedicated CPU
# node (login node OOM-kills concurrent merges). Reuses cc_model_soup_merge.py.
#
# Submit:
#   sbatch --export=ALL,SFT=<inst_dir>,BASE=<base_dir>,ALPHA=0.9,OUT=<out_dir> \
#          slurm/cc_cooldown_soup_merge_cpu.sh
#SBATCH --job-name=cc-cooldown-merge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""

: "${SFT:?need SFT}"
: "${BASE:?need BASE}"
: "${OUT:?need OUT}"
ALPHA="${ALPHA:-0.9}"
PY=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf/bin/python3

echo "Merging alpha=$ALPHA : $SFT (sft) + base $BASE -> $OUT"
"$PY" "$ROOT/scripts/cc_model_soup_merge.py" \
  --sft "$SFT" --base "$BASE" --alpha "$ALPHA" --out "$OUT"

echo "=== result ==="
ls -la "$OUT"/*.safetensors "$OUT"/config.json 2>/dev/null

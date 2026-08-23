#!/usr/bin/env bash
#SBATCH --job-name=fs-qwen3-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python scripts/merge_qwen3_8b_linear.py --alphas 0.25 0.50 0.75 --overwrite

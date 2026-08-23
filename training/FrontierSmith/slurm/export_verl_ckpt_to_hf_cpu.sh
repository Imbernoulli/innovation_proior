#!/usr/bin/env bash
#SBATCH --job-name=fs-export-hf
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-vllm023}"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Python environment not found: ${VENV_DIR}" >&2
  exit 1
fi
source "${VENV_DIR}/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
CKPT_PATH="${CKPT_PATH:-}"

if [ -z "${CKPT_PATH}" ]; then
  if [ -z "${CHECKPOINT_ROOT}" ]; then
    echo "Set CKPT_PATH=/path/to/global_step_N or CHECKPOINT_ROOT=/path/to/checkpoint_root" >&2
    exit 1
  fi

  if [ -f "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]; then
    step="$(tr -d '[:space:]' < "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")"
    CKPT_PATH="${CHECKPOINT_ROOT}/global_step_${step}"
  else
    CKPT_PATH="$(find "${CHECKPOINT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
  fi
fi

if [ -z "${CKPT_PATH}" ] || [ ! -d "${CKPT_PATH}" ]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

if [ -z "${OUTPUT_DIR:-}" ]; then
  root_name="$(basename "$(dirname "${CKPT_PATH}")" | tr -cs 'A-Za-z0-9._-' '_')"
  step_name="$(basename "${CKPT_PATH}" | tr -cs 'A-Za-z0-9._-' '_')"
  OUTPUT_DIR="${PWD}/models/${root_name}_${step_name}_hf"
fi

python scripts/merge_fsdp_to_hf.py \
  --ckpt "${CKPT_PATH}" \
  --output "${OUTPUT_DIR}" \
  --dtype "${DTYPE:-bfloat16}"

echo "EXPORT_OUTPUT_DIR=${OUTPUT_DIR}"

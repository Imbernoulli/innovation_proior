#!/usr/bin/env bash
# Export an already-written VERL checkpoint and run the standard Qwen3 eval.
#
# This is the fallback path for jobs that hit walltime after saving a usable
# global_step_N checkpoint.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

CKPT_PATH="${CKPT_PATH:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
MODEL_TAG="${MODEL_TAG:-}"

if [ -z "$CKPT_PATH" ]; then
  if [ -z "$CHECKPOINT_ROOT" ]; then
    echo "Set CKPT_PATH=/path/to/global_step_N or CHECKPOINT_ROOT=/path/to/checkpoint_root" >&2
    exit 1
  fi
  if [ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]; then
    step="$(tr -d '[:space:]' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")"
    CKPT_PATH="$CHECKPOINT_ROOT/global_step_${step}"
  else
    CKPT_PATH="$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
  fi
fi

if [ -z "$CKPT_PATH" ] || [ ! -d "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

step_name="$(basename "$CKPT_PATH")"
step_num="${step_name#global_step_}"
if [ -z "$MODEL_TAG" ]; then
  root_name="$(basename "$(dirname "$CKPT_PATH")" | tr -cs 'A-Za-z0-9._-' '_')"
  MODEL_TAG="${root_name}_${step_name}"
fi

HF_DIR="${HF_DIR:-$PROJECT_ROOT/models/${MODEL_TAG}_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_${MODEL_TAG}_thinking_general_32k_both_vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_TAG}"
DEPENDENCY="${DEPENDENCY:-}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"

dep_arg=()
if [ -n "$DEPENDENCY" ]; then
  dep_arg=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY}")
fi

echo "checkpoint: $CKPT_PATH"
echo "step: $step_num"
echo "hf_dir: $HF_DIR"
echo "output_dir: $OUTPUT_DIR"

export_text=$(
  sbatch "${dep_arg[@]}" \
    --export="ALL,CKPT_PATH=$CKPT_PATH,OUTPUT_DIR=$HF_DIR" \
    slurm/export_verl_ckpt_to_hf_cpu.sh
)
export_job="${export_text##* }"
echo "export job: $export_job"

eval_text=$(
  sbatch --dependency="afterok:${export_job}" \
    --export="ALL,MODEL_PATH=$HF_DIR,MODEL_TAG=$MODEL_TAG,SERVED_MODEL_NAME=$SERVED_MODEL_NAME,OUTPUT_DIR=$OUTPUT_DIR" \
    slurm/eval_qwen3_both_thinking_1gpu_ailab.sh
)
eval_job="${eval_text##* }"
echo "trained eval job: $eval_job"

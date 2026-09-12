#!/bin/bash
# Unified eval: GSM8K + MATH-500 + AMC23 in one container instance.
# vLLM loads the trained student checkpoint once and reuses it across all 3 splits.
set -euxo pipefail

SAVE_PATH="${SAVE_PATH:-/tmp/distill}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAVE_PATH/student_ckpt}"
CKPT="${OUTPUT_DIR}/student_ckpt"
export CKPT

cd /workspace
python /workspace/_task/scripts/math_eval_all.py

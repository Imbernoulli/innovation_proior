#!/bin/bash
# Train Qwen2.5-0.5B with on-policy distillation against Qwen2.5-Math-7B-Instruct.
# Checkpoint lands at $SAVE_PATH/student_ckpt so the eval_*.sh scripts can load it.

set -euxo pipefail

SEED_VALUE="${SEED:-42}"
export PYTHONHASHSEED="$SEED_VALUE"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
echo "Detected $N_GPUS GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

SAVE_PATH="${SAVE_PATH:-/tmp/distill}"
# OUTPUT_DIR is injected by MLS-Bench as $SAVE_PATH/$task/$exp/seed_$SEED, so it is
# unique per (baseline, seed) and safe under concurrent runs.
OUTPUT_DIR="${OUTPUT_DIR:-$SAVE_PATH/student_ckpt}"
CKPT="${OUTPUT_DIR}/student_ckpt"
mkdir -p "$CKPT"

cd /workspace

# Derive a unique-per-job port. accelerate's --main_process_port=0 is broken:
# the master binds an ephemeral port but the children read port=0 literally and
# hang trying to connect to 127.0.0.1:0. Using the SLURM job ID gives each
# co-scheduled job a distinct port without collisions on the same node.
PORT=$(( 29500 + (${SLURM_JOB_ID:-$$} % 10000) ))
echo "Using main_process_port=$PORT (SLURM_JOB_ID=${SLURM_JOB_ID:-unset})"

accelerate launch \
    --num_processes=$N_GPUS \
    --mixed_precision=bf16 \
    --main_process_port=$PORT \
    /workspace/_task/scripts/train_distill.py \
    --model_name_or_path /data/models/Qwen2.5-0.5B \
    --teacher_model_name_or_path /data/models/Qwen2.5-Math-7B-Instruct \
    --dataset_path /data/datasets/openr1_math_10k.parquet \
    --output_dir "$CKPT" \
    --seed "$SEED_VALUE" \
    --max_steps 300 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_new_tokens 768 \
    --beta 0.5 \
    --lmbda 0.5 \
    --temperature 0.9 \
    --logging_steps 25

ls -la "$CKPT"

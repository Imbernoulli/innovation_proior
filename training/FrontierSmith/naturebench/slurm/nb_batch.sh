#!/bin/bash
# NatureBench subset eval — one Slurm job = one shard (vLLM loaded once, N tasks).
#
# Submit (gpu-ee, the uncontended lane):
#   cd naturebench && sbatch --export=ALL,BATCH=nb_9b,MODEL_PATH=...,SHARD=0,NSHARDS=2 \
#        slurm/nb_batch.sh
# Or PLI H100:
#   sbatch -p pli --account=goedelprover --qos=pli-low --gres=gpu:h100:1 \
#        --export=ALL,BATCH=...,SHARD=0,NSHARDS=2 slurm/nb_batch.sh
#
# NOTE: task setup (pip) must already be done on a login node
#       (`nb_setup_all.sh`) — compute nodes have no outbound DNS.

#SBATCH --job-name=nb_batch
#SBATCH --partition=gpu-ee
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/naturebench/logs/nbbatch-%j.out

set -uo pipefail

NB=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/naturebench
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

BATCH="${BATCH:?set BATCH}"
TASKS_FILE="${TASKS_FILE:-$NB/task-sets/working.txt}"
MODEL_PATH="${MODEL_PATH:-$FS/models/Qwen3.5-9B}"
SERVED_NAME="${SERVED_NAME:-nb-model}"
SHARD="${SHARD:-0}"
NSHARDS="${NSHARDS:-1}"
MODE="${MODE:-agent}"
TIMEOUT="${TIMEOUT:-1500}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TASK_GPU_FLAG="${TASK_GPU_FLAG:---task-gpu}"   # set empty to deny tasks the GPU

export APPTAINER_CACHEDIR="$NB/containers/.apptainer-cache"
export APPTAINER_TMPDIR="$NB/containers/.apptainer-tmp"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "[nb_batch] batch=$BATCH shard=$SHARD/$NSHARDS model=$MODEL_PATH node=$(hostname)"
nvidia-smi -L 2>/dev/null | head -2

python "$NB/harness/nb_batch.py" \
    --batch "$BATCH" \
    --tasks-file "$TASKS_FILE" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --model-path "$MODEL_PATH" \
    --served-name "$SERVED_NAME" \
    --mode "$MODE" \
    --timeout "$TIMEOUT" \
    --max-rounds "$MAX_ROUNDS" \
    --run-timeout "$RUN_TIMEOUT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    $TASK_GPU_FLAG

echo "[nb_batch] exit=$?"

#!/usr/bin/env bash
# Export a VERL FSDP checkpoint to HuggingFace format, then evaluate it through
# a vLLM OpenAI-compatible server on FrontierCS + ALE-Bench.
# Required when the training job has produced at least one global_step_* dir.

#SBATCH --job-name=fs-q35-ckpt-eval
#SBATCH --partition=ailab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=420G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"  # HARDCODED (spool/submit-dir pitfall)
cd "$PROJECT_ROOT"
mkdir -p logs

export VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-vllm023}"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="/tmp"

export ALE_BENCH_DATA="$PROJECT_ROOT/data/alebench/local_data"
export ALE_BENCH_CACHE="$PROJECT_ROOT/.cache/ale-bench"
export ALE_BENCH_TOOL_CACHE="$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds"
export ALE_BENCH_CONTAINER_BACKEND=apptainer
export ALE_BENCH_APPTAINER_DIR="$PROJECT_ROOT/.cache/apptainer/alebench"
export ALEBENCH_JUDGE_VERSION=202301
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-4}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b/qwen35_9b_grpo}"
CKPT_PATH="${CKPT_PATH:-}"

if [ -z "$CKPT_PATH" ]; then
  if [ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]; then
    step="$(tr -d '[:space:]' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")"
    CKPT_PATH="$CHECKPOINT_ROOT/global_step_$step"
  else
    CKPT_PATH="$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
  fi
fi

if [ -z "$CKPT_PATH" ] || [ ! -d "$CKPT_PATH" ]; then
  echo "Checkpoint not found. Set CKPT_PATH or CHECKPOINT_ROOT after training writes global_step_*." >&2
  exit 1
fi

step_name="$(basename "$CKPT_PATH" | tr -cs 'A-Za-z0-9._-' '_')"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_ROOT/models/qwen35_9b_grpo_${step_name}_hf}"

if ! find "$MODEL_OUTPUT_DIR" -maxdepth 1 \( -name 'model.safetensors' -o -name 'model-*.safetensors' \) -print -quit 2>/dev/null | grep -q .; then
  python scripts/merge_fsdp_to_hf.py \
    --ckpt "$CKPT_PATH" \
    --output "$MODEL_OUTPUT_DIR" \
    --dtype "${DTYPE:-bfloat16}"
else
  echo "Using existing exported model: $MODEL_OUTPUT_DIR"
fi

export PORT="${PORT:-8082}"
export GJ_PORT="${GJ_PORT:-5050}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-ckpt-eval-${SLURM_JOB_ID}}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

export VLLM_PORT="${VLLM_PORT:-8000}"
export HOST=127.0.0.1
export MODEL_PATH="$MODEL_OUTPUT_DIR"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b-${step_name}}"
export TP="${TP:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-26624}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
PORT="$VLLM_PORT" scripts/start_vllm_server.sh &
VLLM_PID="$!"

cleanup() {
  kill "$VLLM_PID" "$JUDGE_PID" >/dev/null 2>&1 || true
  wait "$VLLM_PID" "$JUDGE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "$JUDGE_PID" >/dev/null 2>&1; then
    echo "FrontierCS judge launcher exited early" >&2
    exit 1
  fi
done
curl -fsS "http://127.0.0.1:${PORT}/health"
echo

for _ in $(seq 1 720); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM server exited early" >&2
    exit 1
  fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models"
echo

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval_qwen35_9b_grpo_${step_name}_vllm}" \
N_SAMPLES="${N_SAMPLES:-5}" \
MAX_TOKENS="${MAX_TOKENS:-16000}" \
CONCURRENCY="${CONCURRENCY:-32}" \
ENABLE_THINKING="${ENABLE_THINKING:-0}" \
VLLM_PORT="$VLLM_PORT" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
bash scripts/eval_base_model_qwen35_9b_vllm_request.sh "$@"

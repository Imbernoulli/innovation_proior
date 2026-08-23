#!/usr/bin/env bash
# Validation-only Qwen3.5-9B base evaluation on FrontierCS + ALE-Bench-lite.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "ERROR: .venv not found. Run setup-env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$PROJECT_ROOT/.cache/vllm}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-base-eval}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1

export ALE_BENCH_DATA="${ALE_BENCH_DATA:-$PROJECT_ROOT/data/alebench/local_data}"
export ALE_BENCH_CACHE="${ALE_BENCH_CACHE:-$PROJECT_ROOT/.cache/ale-bench}"
export ALE_BENCH_TOOL_CACHE="${ALE_BENCH_TOOL_CACHE:-$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds}"
export ALE_BENCH_CONTAINER_BACKEND="${ALE_BENCH_CONTAINER_BACKEND:-apptainer}"
export ALE_BENCH_APPTAINER_DIR="${ALE_BENCH_APPTAINER_DIR:-$PROJECT_ROOT/.cache/apptainer/alebench}"
export ALEBENCH_JUDGE_VERSION="${ALEBENCH_JUDGE_VERSION:-202301}"
export ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"
FRONTIERCS_VAL_DATA="${FRONTIERCS_VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}"
ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}"

NGPU="${NGPU:-2}"
TP="${TP:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-26624}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-26624}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_base_frontiercs_alebench_val}"
PROJECT_NAME="${PROJECT_NAME:-frontiersmith_qwen35_9b_base_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/base_eval_qwen35_9b}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints/base_eval_qwen35_9b}"
VALIDATION_DIR="${VALIDATION_DIR:-$OUTPUT_DIR/validation_data}"

for path in "$MODEL_PATH" "$FRONTIERCS_VAL_DATA" "$ALEBENCH_VAL_DATA" "$ALE_BENCH_DATA" "$ALE_BENCH_APPTAINER_DIR" "$ALE_BENCH_TOOL_CACHE"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$VALIDATION_DIR"

TRAIN_FILES="['$FRONTIERCS_VAL_DATA']"
VAL_FILES="['$FRONTIERCS_VAL_DATA','$ALEBENCH_VAL_DATA']"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size=8 \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.prompt_key=prompt \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enforce_eager=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.val_kwargs.n=5 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward.num_workers="$REWARD_NUM_WORKERS" \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.validation_data_dir="$VALIDATION_DIR" \
  trainer.val_only=True \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  "$@"

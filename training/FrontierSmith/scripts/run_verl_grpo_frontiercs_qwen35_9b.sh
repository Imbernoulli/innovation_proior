#!/bin/bash
# GRPO training for Qwen3.5-9B on Frontier-CS (VERL)
#
# Prerequisites:
#   1. create .venv-vllm023 with vLLM >= 0.23.0 and VERL deps
#   2. python scripts/prepare_frontiercs_parquet.py
#   3. Frontier-CS judge on :8082 (cd Frontier-CS/algorithmic && ./run_judge.sh)
#   4. (optional) python scripts/prepare_alebench_parquet.py --no-lite
#   5. 2+ GPUs (default 4 H200 GPUs; set NGPU/TP explicitly for other layouts)
#
# Usage:
#   bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh
#   NGPU=4 TP=1 bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh
#
# Start from scratch (backup existing checkpoints, do not delete):
#   FRESH_START=1 bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh
#
# To debug response length spikes, add:
#   trainer.dump_long_responses_dir=outputs/dump_long_responses \
#   trainer.dump_long_responses_threshold=20000 \

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Environment ──────────────────────────────────────────────────────────────
# Activate the Qwen3.5-capable venv; vLLM 0.8.x cannot load Qwen3.5.
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-vllm023}"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: $VENV_DIR not found. Build the vLLM 0.23 training environment first."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$PROJECT_ROOT/.cache/vllm}"
unset VLLM_USE_V1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export ALE_BENCH_CACHE_DIR="${ALE_BENCH_CACHE_DIR:-$PROJECT_ROOT/.cache/ale-bench}"
export ALE_BENCH_DATA="${ALE_BENCH_DATA:-$PROJECT_ROOT/data/alebench/local_data}"
export ALE_BENCH_CACHE="${ALE_BENCH_CACHE:-$PROJECT_ROOT/.cache/ale-bench}"
export ALE_BENCH_TOOL_CACHE="${ALE_BENCH_TOOL_CACHE:-$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds}"
export ALE_BENCH_REQUIRE_TOOL_CACHE="${ALE_BENCH_REQUIRE_TOOL_CACHE:-1}"
export ALE_BENCH_CONTAINER_BACKEND="${ALE_BENCH_CONTAINER_BACKEND:-apptainer}"
export ALE_BENCH_APPTAINER_DIR="${ALE_BENCH_APPTAINER_DIR:-$PROJECT_ROOT/.cache/apptainer/alebench}"
export ALEBENCH_JUDGE_VERSION="${ALEBENCH_JUDGE_VERSION:-202301}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}

TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}
ALEBENCH_VAL_DATA=${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}

TP=${TP:-1}
NGPU=${NGPU:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-26624}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-True}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-5}
# Train-rollout sampling (defaults = stock verl: pure sampling). Set to match the
# eval protocol (top_p 0.95, top_k 20, + ROLLOUT_PRESENCE_PENALTY=1.5 via env,
# plumbed in agent_loop.py) when the arm should optimize the evaluated behavior.
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-60}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
# Checkpoint rotation: keep at most this many actor checkpoints on disk (older
# global_step_* dirs are pruned by verl's checkpoint manager as new ones land).
# Empty/unset => omit the flag => verl default (null = keep ALL). Disk-safety knob
# for the amplify expansion; leaving it empty keeps existing/running jobs unchanged.
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
PROJECT_NAME=${PROJECT_NAME:-verl_frontiercs_qwen35_9b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_grpo}

# ── Throughput / memory knobs (overridable; defaults preserve the proven recipe) ──
# The proven 4-GPU recipe ran with full ACTOR CPU offload (param+optimizer) which made
# update_actor ~700s/step. On 4xH200 FSDP already shards optimizer/params across GPUs,
# so offload is unnecessary. The rlm_amplify_v3 launcher sets ACTOR_*_OFFLOAD=False,
# raises ppo_micro_batch_size, and lowers vLLM gpu_memory_utilization to make room.
# Defaults below = the OLD proven values, so anything NOT passing these is unchanged.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
# Long-sequence memory controls (default OFF => proven 9B recipe unchanged).
# For 35B: set USE_DYNAMIC_BSZ=True (token-budget micro-batching instead of
# fixed 1-seq/micro-batch) + ULYSSES_SP=2 (shard each long sequence across 2
# GPUs along the seq dim => per-GPU backward activation ~halved, keeps RESP=32768).
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ULYSSES_SP=${ULYSSES_SP:-1}
# torch.distributed process-group timeout (2026-08-05). verl defaults to 600s
# (fsdp_workers.py:161) and the NCCL_TIMEOUT env var does NOT feed it. A single
# rollout step at 64x16 with 32k generations takes >1h, so the post-generation
# collective blew the 600s watchdog and SIGABRT-ed all 8 ranks 2h05m into the
# run (job 12024996). 7200s covers a slow step with margin.
NCCL_PG_TIMEOUT=${NCCL_PG_TIMEOUT:-7200}
# Entropy over a 248320-token vocab materializes ppo_max_token_len_per_gpu x vocab
# in fp32: at 29696 tokens that is 27.5 GiB in ONE allocation, which OOMed the
# 4-GPU arm (job 12064548) -- 4-way FSDP leaves twice the per-GPU model state of
# the 8-way arm, so the spike no longer fits. Chunking computes it in slices.
ENTROPY_CHUNKING=${ENTROPY_CHUNKING:-True}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
# torch.compile toggle. Default True = proven 9B (static-shape) recipe. For 35B with
# USE_DYNAMIC_BSZ=True the micro-batch shapes vary every step; torch.compile
# (applied dynamic=True to entropy/log-prob helpers, dp_actor.py:87/104) can hit
# guard failures / recompilation stalls that hang a straggler rank right at the
# step-1 checkpoint barrier (smoke 11266326/11270526: save entered, dir created,
# then silent hang after Inductor "online softmax" compile warnings). Set False for 35B.
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-True}
# HOST-RAM cap for Ray's plasma object store (bytes). Empty => Ray default (~30%
# of cgroup ≈ 435G on a 1450G node). For 35B the checkpoint save's offload_to_cpu
# spike stacked on optim(288G)+ref offload OOM-killed a worker at save time
# (node is 1464G physical, we ask 1450G => ~14G margin). Rollout data through
# plasma is small (~1G), so capping the store to ~50G frees ~385G for the save.
RAY_OBJECT_STORE_BYTES=${RAY_OBJECT_STORE_BYTES:-}

# Ensure data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Run: python scripts/prepare_frontiercs_parquet.py"
    exit 1
fi

TRAIN_FILES="['$TRAIN_DATA']"

# Validate against Frontier-CS + ALE-Bench (when present).
VAL_LIST=()
if [ -f "$VAL_DATA" ]; then
    VAL_LIST+=("'$VAL_DATA'")
else
    VAL_LIST+=("'$TRAIN_DATA'")
fi
if [ -f "$ALEBENCH_VAL_DATA" ]; then
    VAL_LIST+=("'$ALEBENCH_VAL_DATA'")
    echo "[ALE-Bench] Including $ALEBENCH_VAL_DATA in validation."
else
    echo "[ALE-Bench] $ALEBENCH_VAL_DATA not found; run scripts/prepare_alebench_parquet.py to enable."
fi
VAL_FILES="[$(IFS=,; echo "${VAL_LIST[*]}")]"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── Checkpoints & outputs (all in project dir) ──────────────────────────────
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b/qwen35_9b_grpo}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rollout_data}"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

FRESH_START_ARGS=""
if [ "${FRESH_START:-0}" = "1" ]; then
    if [ -d "$CKPT_DIR" ]; then
        BACKUP_DIR="${CKPT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[FRESH_START] Backing up checkpoints: $CKPT_DIR -> $BACKUP_DIR"
        mv "$CKPT_DIR" "$BACKUP_DIR"
        echo "[FRESH_START] Backup done."
    fi
    FRESH_START_ARGS="trainer.resume_mode=disable"
    echo "[FRESH_START] Training will start from scratch (resume_mode=disable)."
fi

# ── Actor attention implementation ──────────────────────────────────────────
# Standard acceleration: use flash_attention_2 for the FSDP2 actor when the
# flash_attn package is importable in this venv (another track is building it).
# vLLM rollout already uses FLASH_ATTN (see VLLM_ENGINE_ARGS below). If flash_attn
# is NOT importable we fall back to sdpa (the previous working setting).
# Override explicitly with ACTOR_ATTN_IMPL=flash_attention_2|sdpa.
if [ -z "${ACTOR_ATTN_IMPL:-}" ]; then
    if python -c "import flash_attn" >/dev/null 2>&1; then
        ACTOR_ATTN_IMPL="flash_attention_2"
        echo "[attn] flash_attn importable -> actor attn_implementation=flash_attention_2"
    else
        ACTOR_ATTN_IMPL="sdpa"
        echo "[attn] flash_attn NOT importable -> actor attn_implementation=sdpa" >&2
    fi
fi
# 2026-08-05 (user directive): training MUST use flash attention. A silent fall back
# to sdpa costs a large fraction of throughput at 32k sequence length and is exactly
# the class of quiet degradation this project keeps getting burned by. Abort instead.
# Escape hatch for a deliberate sdpa run: ALLOW_SDPA=1.
if [ "$ACTOR_ATTN_IMPL" = "sdpa" ] && [ "${ALLOW_SDPA:-0}" != "1" ]; then
    echo "FATAL: flash_attn is not importable in $VENV_DIR -- refusing to train on sdpa." >&2
    echo "       Install the prebuilt wheel (no compilation needed, login node has network):" >&2
    echo "       pip install --no-deps .cache/wheels/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" >&2
    echo "       Override intentionally with ALLOW_SDPA=1." >&2
    exit 1
fi

if [ -z "${GDN_PREFILL_BACKEND+x}" ]; then
    GDN_PREFILL_BACKEND="$(
        MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os
from pathlib import Path

try:
    model_type = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text()).get("model_type")
except Exception:
    model_type = None

print("triton" if model_type in {"qwen3_5", "qwen3_5_moe"} else "")
PY
    )"
fi

VLLM_ENGINE_ARGS=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE"
)
if [ -n "$GDN_PREFILL_BACKEND" ]; then
    VLLM_ENGINE_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=$GDN_PREFILL_BACKEND")
fi

# ---- DAPO-style anti-collapse knobs (env-gated; 9B defaults reproduce vanilla GRPO) ----
# Rationale (35B synth RL forensics, 2026-07-20): vanilla GRPO death-spiralled —
# reward inverted-U (peak s5 ~0.23 -> s20 ~0.08), response len collapsed 26k->9k
# (losers longer than winners -> policy shortens -> too shallow for hard problems),
# all-zero groups rose 0%->31% (gradient starvation), entropy crashed s17-20 (lock-in).
# Fixes: clip-higher (entropy), higher response cap (truncation), optional entropy
# bonus / KL drop. NOTE: filter_groups (dynamic sampling) is STILL NOT implemented in
# the main_ppo RayPPOTrainer (only in experimental/fully_async_policy) -> enabling it
# here is a silent no-op; left wired for a future port.
#
# The dead-group problem it was meant to address is now handled by two env-gated
# features that live outside this arg list (they are read from the environment, not
# from hydra, so nothing needs to be threaded through here):
#   ADAPTIVE_N_ENABLE=1   per-prompt adaptive resampling -- deepen a flat group on the
#                         SAME prompt (n -> 2n -> ... -> ADAPTIVE_N_MAX) rather than
#                         swapping in a fresh prompt the way filter_groups would.
#                         See verl/trainer/ppo/adaptive_sampling.py.
#   FS_OVERLONG_PENALTY=1 DAPO soft overlong punishment on the agent-loop path.
#                         See verl/utils/reward_score/overlong_penalty.py.
# Defaults for both are OFF, i.e. vanilla GRPO.
CLIP_LOW="${CLIP_LOW:-0.2}"
CLIP_HIGH="${CLIP_HIGH:-0.2}"                 # DAPO: 0.28
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"         # DAPO: 0.0
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"           # optional small bonus, e.g. 0.001
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
FILTER_GROUPS_MAX_GEN_BATCHES="${FILTER_GROUPS_MAX_GEN_BATCHES:-10}"
DAPO_ARGS=()
if [ "$FILTER_GROUPS_ENABLE" = "True" ]; then
    DAPO_ARGS+=(
        "+algorithm.filter_groups.enable=True"
        "+algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC"
        "+algorithm.filter_groups.max_num_gen_batches=$FILTER_GROUPS_MAX_GEN_BATCHES"
    )
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=$ACTOR_ATTN_IMPL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.nccl_timeout=$NCCL_PG_TIMEOUT \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=$ENTROPY_CHUNKING \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-token-mean} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.use_torch_compile=$USE_TORCH_COMPILE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SP \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ULYSSES_SP \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    "${VLLM_ENGINE_ARGS[@]}" \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    algorithm.use_kl_in_reward=False \
    "${DAPO_ARGS[@]}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    ${MAX_ACTOR_CKPT_TO_KEEP:+trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP} \
    ${RAY_OBJECT_STORE_BYTES:+ +ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_BYTES} \
    $FRESH_START_ARGS \
    "$@"

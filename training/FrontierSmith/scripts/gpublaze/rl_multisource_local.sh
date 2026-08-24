#!/usr/bin/env bash
# gpublaze RL launcher: multisource GRPO, 4B / 2xH100 recipe.
# Local (no-slurm) port of slurm/cc_rl_multisource.sh -- same env-knob surface,
# same runner (scripts/run_verl_grpo_frontiercs_qwen35_9b.sh via snapshot), with
# the machine deltas confined here:
#   * mix = frontiersmith_synth + FCS algorithmic (train_synth_fcs.parquet,
#     y26 pure-time-sentence conditioning); NO mlsbench rows (no _tv variants
#     on this box), NO research rows (2026-08-04: eval-only)
#   * FCS reward -> the standing local judge (FRONTIERCS_JUDGE_URL, :8082;
#     autostarted if down)
#   * venv = .venv-rl-gpublaze (vllm 0.21 / torch 2.11 cu128 / fla stack)
#   * 4B/2-GPU shapes derived from the 9B/4-GPU blueprint (see comments)
#
# Smoke (1 step, no ckpt, GPU 7 -- coordinator-cleared 2026-08-23):
#   GPUS=7 FS_ALLOW_GPU67=1 NGPU=1 TOTAL_TRAINING_STEPS=1 ROLLOUT_N=2 \
#   TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 SAVE_FREQ=100000 \
#   TRAIN_DATA=$PWD/data/multisource_rl/train_synth_fcs_smoke.parquet \
#   EXPERIMENT_NAME=ms_smoke_gpublaze bash scripts/gpublaze/rl_multisource_local.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_gpublaze.sh"
PROJECT_ROOT="$FS_ROOT"
cd "$PROJECT_ROOT"
mkdir -p logs

GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5 (2 GPUs) or GPUS=7 (1-GPU smoke)}"
fs_guard_gpus "$GPUS" || exit 1
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU_DEFAULT=$(awk -F, '{print NF}' <<<"$GPUS")
export NGPU="${NGPU:-$NGPU_DEFAULT}"

# ---- node guards (blueprint minus slurm/module machinery) ---------------------
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FS_VLLM_PENALTY_FASTPATH="${FS_VLLM_PENALTY_FASTPATH:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ulimit -s 8192 2>/dev/null || true

# ---- reward env: synth (bwrap sandbox, local corpus) --------------------------
export FRONTIERSMITH_SYNTH_ROOT="${FRONTIERSMITH_SYNTH_ROOT:-/srv/home/bohanlyu/innovation_proior/frontiersmith_synth}"
export FRONTIERSMITH_SYNTH_FAIL_SOFT="${FRONTIERSMITH_SYNTH_FAIL_SOFT:-1}"
export FRONTIERSMITH_SYNTH_MAX_WALL="${FRONTIERSMITH_SYNTH_MAX_WALL:-300}"
export FRONTIERSMITH_SYNTH_MAX_CONC="${FRONTIERSMITH_SYNTH_MAX_CONC:-4}"
export FSX_CHILD_MEM_MB="${FSX_CHILD_MEM_MB:-8192}"
export MLS_RL_EPISODE_MEM_MB="${MLS_RL_EPISODE_MEM_MB:-16384}"   # kept (3-layer guard), inert without MLS rows
# research env kept form-compatible (inert: no research rows in the mix)
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-$FS_ROOT/.venv-gpublaze/bin/python}"
export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-64}"
export FRONTIERCS_RESEARCH_CPU_TIMEOUT="${FRONTIERCS_RESEARCH_CPU_TIMEOUT:-300}"

# ---- reward env: FCS algorithmic -> local judge -------------------------------
export FRONTIERCS_JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://127.0.0.1:8082}"
if ! curl -fsS --max-time 5 "$FRONTIERCS_JUDGE_URL/health" >/dev/null 2>&1; then
  if [ "${FS_AUTOSTART_JUDGE:-1}" = "1" ]; then
    echo "[rl-local] judge not answering; autostarting on :8082"
    PORT=8082 GJ_PORT=5050 GJ_BACKEND=gojudge GJ_PARALLELISM=16 JUDGE_WORKERS=16 \
    RUNTIME_DIR="$PROJECT_ROOT/.cache/frontiercs-judge/8082" \
      setsid nohup bash "$SCRIPT_DIR/start_frontiercs_judge_local.sh" >> "$PROJECT_ROOT/logs/judge_8082.log" 2>&1 &
    for _ in $(seq 1 60); do curl -fsS "$FRONTIERCS_JUDGE_URL/health" >/dev/null 2>&1 && break; sleep 2; done
  fi
  curl -fsS "$FRONTIERCS_JUDGE_URL/health" >/dev/null 2>&1 \
    || { echo "FATAL: FCS judge unreachable at $FRONTIERCS_JUDGE_URL -- every frontiercs reward would fail" >&2; exit 1; }
fi

# ---- per-task reward norm (0-100 -> [0,1]) ------------------------------------
export FS_PERTASK_REWARD_NORM="${FS_PERTASK_REWARD_NORM:-1}"
export FS_PERTASK_REWARD_NORM_TARGET="${FS_PERTASK_REWARD_NORM_TARGET:-1.0}"

# ---- anti-dead-group knobs: blueprint defaults (all OFF = vanilla GRPO) -------
export ADAPTIVE_N_ENABLE="${ADAPTIVE_N_ENABLE:-0}"
export FS_OVERLONG_PENALTY="${FS_OVERLONG_PENALTY:-0}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"

# ---- data ---------------------------------------------------------------------
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/multisource_rl/train_synth_fcs.parquet}"
export VAL_DATA="${VAL_DATA:-$TRAIN_DATA}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/__none__.parquet}"
[ -f "$TRAIN_DATA" ] || { echo "ERROR: $TRAIN_DATA missing. Build: .venv-gpublaze/bin/python scripts/gpublaze/prepare_synth_fcs_rl_parquet.py" >&2; exit 1; }

# ---- schedule -----------------------------------------------------------------
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ms_qwen35_4b_grpo_gpublaze}"
# Interactive problems + 16-way rollout bursts can queue past the scorer's 900s
# default at the standing judge -> spurious JudgeInfraError -> unfair 0s (seen
# on problem 153, 2026-08-23). Raise the wait; queue delay is not solution quality.
export FRONTIERCS_JUDGE_MAX_WAIT="${FRONTIERCS_JUDGE_MAX_WAIT:-2400}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-60}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export PROJECT_NAME="${PROJECT_NAME:-rl_multisource_gpublaze}"
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/rl_multisource/${EXPERIMENT_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rl_multisource_rollout/${EXPERIMENT_NAME}}"

# ---- GRPO shape (blueprint 9B/4-GPU values kept; halve only what must halve) --
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export VAL_N="${VAL_N:-1}"

# ---- memory/speed, 4B on 2x80G (blueprint was 9B on 4xH200-141G) --------------
# vLLM co-located on 80G cards: 0.5 leaves ~40G for the FSDP actor phase.
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"

# ---- context: no MLS rows -> prompt budget shrinks 26624 -> 10240 (FCS max).
# verl packing assert: PPO_MAX_TOKEN_LEN_PER_GPU x ULYSSES_SP >= prompt+response.
# 43008 x 1 = 10240 + 32768. Generation floor 32K is a standing user rule.
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
export ROLLOUT_PRESENCE_PENALTY="${ROLLOUT_PRESENCE_PENALTY:-1.5}"
export ROLLOUT_MIN_P="${ROLLOUT_MIN_P:-0.0}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-43008}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-43008}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"   # 8 was the Princeton 9B tight-memory inheritance; on 4B it serialized 1024 rollouts into waves of 8 (2.9h steps). vLLM admits by KV headroom, 256 is a cap not a reservation.
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-43008}"
export ULYSSES_SP="${ULYSSES_SP:-1}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"

export TP="${TP:-1}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"

# ---- model: resolve the HF-cache snapshot to a real dir (the runner probes
# $MODEL_PATH/config.json for the qwen3_5 GDN prefill backend) -----------------
if [ -z "${MODEL_PATH:-}" ]; then
  MODEL_PATH="$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ 2>/dev/null | head -1)"
  MODEL_PATH="${MODEL_PATH%/}"   # verl asserts no trailing slash
fi
[ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] \
  || { echo "FATAL: MODEL_PATH unresolved (want the Qwen3.5-4B snapshot dir); got '$MODEL_PATH'" >&2; exit 1; }
export MODEL_PATH

# ---- venv + caches ------------------------------------------------------------
export VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-rl-gpublaze}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-gpublaze-$USER}"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR" "$RAY_TMPDIR"

echo "[rl-local] TRAIN_DATA=$TRAIN_DATA"
echo "[rl-local] MODEL=$MODEL_PATH GPUS=$GPUS NGPU=$NGPU TP=$TP steps=$TOTAL_TRAINING_STEPS batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N}"
echo "[rl-local] judge=$FRONTIERCS_JUDGE_URL synth_root=$FRONTIERSMITH_SYNTH_ROOT"
echo "[rl-local] ctx: prompt=$MAX_PROMPT_LENGTH resp=$MAX_RESPONSE_LENGTH model_len=$MAX_MODEL_LEN gpu_util=$GPU_MEMORY_UTILIZATION"

# ---- sandbox preflight (synth rows FAIL_SOFT to silent 0.0 without bwrap) -----
if [ "${SKIP_BWRAP_PREFLIGHT:-0}" != "1" ]; then
  bwrap --ro-bind / / true 2>/dev/null \
    || { echo "FATAL: bwrap dead on $(hostname); synth rewards would all FAIL_SOFT to 0" >&2; exit 1; }
  echo "[rl-local] sandbox backend: bwrap"
fi

# ---- runner snapshot (blueprint convention: one level under PROJECT_ROOT) -----
RUNNER_SNAPSHOT="${RUNNER_SNAPSHOT:-$PROJECT_ROOT/.runner_snapshots/run_verl_grpo_local_$$.sh}"
mkdir -p "$(dirname "$RUNNER_SNAPSHOT")"
cp scripts/run_verl_grpo_frontiercs_qwen35_9b.sh "$RUNNER_SNAPSHOT"
echo "[rl-local] runner snapshot: $RUNNER_SNAPSHOT"

exec bash "$RUNNER_SNAPSHOT" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/config/mlsbench_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers="${MLS_RL_AGENT_WORKERS:-2}" \
    "$@"

#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
cd "${WORK_ROOT}"

# Triton 3.0.0, used under vLLM 0.6.3 kernels, ignores LD_LIBRARY_PATH when
# ldconfig reports a stale CUDA compat entry. Point Triton's libcuda search
# directly at the Apptainer --nv driver bind mount.
if [ -e /.singularity.d/libs/libcuda.so ]; then
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/.singularity.d/libs}"
fi
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

TASK_DIR="${TASK_DIR:-${WORK_ROOT}/_task}"
PKG_DIR="${PKG_DIR:-${WORK_ROOT}/Unify-Post-Training}"
OPENR1_PATH="${OPENR1_PATH:-/root/data/openr1.parquet}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen2.5-Math-1.5B}"
SEED_VALUE="${SEED:-42}"
if [ -n "${OUTPUT_DIR:-}" ]; then
  RUN_ROOT="${OUTPUT_DIR}"
elif [ -n "${MLSBENCH_PKG_DIR:-}" ]; then
  RUN_ROOT="$(dirname "${MLSBENCH_PKG_DIR}")/_outputs/seed_${SEED_VALUE}"
else
  RUN_ROOT="/tmp/mlsbench_hpt_shared/seed_${SEED_VALUE}"
  echo "WARNING: OUTPUT_DIR is not set; using ${RUN_ROOT}. Set OUTPUT_DIR for concurrent or relay runs." >&2
fi
RUN_NAME="$(basename "$(dirname "${RUN_ROOT}")")"
IS_HPT_BASELINE="${IS_HPT_BASELINE:-}"
if [ -z "${IS_HPT_BASELINE}" ]; then
  if [ "${RUN_NAME}" = "hpt" ]; then
    IS_HPT_BASELINE=1
  else
    IS_HPT_BASELINE=0
  fi
fi
MODEL_RUN_DIR="${RUN_ROOT}/Qwen2.5-Math-1.5B-ctx16k"
PKG_DATA_SUBDIR="data"
PKG_DATA_DIR="${PKG_DIR}/${PKG_DATA_SUBDIR}"
SPLIT_DIR="${RUN_ROOT}/shared_splits"
TRAIN_LOG="${RUN_ROOT}/shared_train.log"
FINAL_METRICS="${RUN_ROOT}/final_metrics.txt"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
TRAIN_SIZE="${TRAIN_SIZE:-25600}"
TRAIN_SOURCE_SPEC="${TRAIN_SOURCE_SPEC:-}"
TRAIN_SOURCE_SELECTION="${TRAIN_SOURCE_SELECTION:-random}"
EVAL_AIME24_PATH="${EVAL_AIME24_PATH:-${PKG_DATA_DIR}/AIME24/test.parquet}"
EVAL_AMC23_PATH="${EVAL_AMC23_PATH:-${PKG_DATA_DIR}/AMC23/test.parquet}"
EVAL_MATH500_PATH="${EVAL_MATH500_PATH:-${PKG_DATA_DIR}/MATH-500/test.parquet}"
AIME24_SIZE="${AIME24_SIZE:-30}"
AMC23_SIZE="${AMC23_SIZE:-40}"
MATH500_SIZE="${MATH500_SIZE:-500}"
INTERNAL_AIME24_SIZE="${INTERNAL_AIME24_SIZE:-1}"
INTERNAL_AMC23_SIZE="${INTERNAL_AMC23_SIZE:-1}"
INTERNAL_MATH500_SIZE="${INTERNAL_MATH500_SIZE:-1}"
# HPT uses the SAME train batch as the sft/grpo baselines (128). The earlier
# 32 was a memory workaround for the *soft* strategy, which keeps all on-policy
# AND adds off-policy SFT (~2× samples vs GRPO). We now use *switch* (the paper
# method), which REMOVES on-policy when it adds off-policy, so its per-step
# footprint is <= the pure-GRPO baseline's — and GRPO ran fine at bsz=128. This
# keeps hpt directly comparable to sft/grpo (identical compute; only the router
# differs).
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_N_GPUS="${EVAL_N_GPUS:-1}"
EXTERNAL_EVAL_SAMPLES="${EXTERNAL_EVAL_SAMPLES:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
# Same mini batch as sft/grpo (bsz/mini == 4); the earlier HPT-specific 16 went
# with the old bsz=32 workaround that we've now removed.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
# Same per-forward microbatch (2) as sft/grpo. With switch (footprint <= GRPO)
# and gpu_mem=0.40, micro=2 fits; the earlier micro=1 went with the bsz=32 path.
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
# SFT loss weight for the off-policy demos. sft/grpo use 1.0; the HPT branch
# overrides to 0.3 (the reference's Qwen2.5-Math-1.5B value) to curb over-SFT.
SFT_LOSS_COEF="${SFT_LOSS_COEF:-1.0}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-4}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-12288}"
LOG_PROB_USE_DYNAMIC_BSZ="${LOG_PROB_USE_DYNAMIC_BSZ:-True}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
# HPT now uses the SAME runtime as sft/grpo: TP=2, n=4, verify_n=4, with the
# paper's 'switch' router (exp_scripts/train.sh). The old TP=1 + 'soft' combo was
# a workaround for a contiguity crash: the off-policy rollout path stored a
# non-contiguous `prompts` (repeat_interleave) that fsdp_vllm.postprocess_data
# broadcast WITHOUT .contiguous(), so TP>1 raised "Tensors must be contiguous".
# That is now fixed package-wide in pre_edit.py (data.batch.contiguous() before
# the broadcast), so TP=2 is safe — and the broadcast contiguity check fires
# even at TP=1, so TP=1 was never a real fix. 'switch' removes on-policy when it
# adds off-policy SFT (footprint <= GRPO), and the prior ZeroDiv came from the
# tiny advantage-split groups (n_verify=2), now guarded by _GRPO_SPLIT_GUARD and
# n_verify=4. ROLLOUT_TP_SIZE_HPT kept as an override knob, defaulting to 2.
#
# HPT also forces use_dynamic_bsz=True (token-budgeted micro-batches, capped at
# ppo_max_token_len_per_gpu) — this is what the paper reference (exp_scripts/
# train.sh) uses for switch. The switch batch MIXES short on-policy rollouts
# (<=4096 tok) with long off-policy SFT demos (prefix up to max_prefix_len=8192).
# With the fixed ppo_micro_batch_size=2 (use_dynamic_bsz=False) a micro-batch can
# pair two long demos → unbounded actor-forward activation/logit spike that
# abruptly killed one rank inside update_actor (job 9130478 died there with no
# CUDA-OOM line, host RAM only 57/400G, then the other ranks hung 600s on the
# next NCCL collective). Dynamic bsz bounds per-GPU tokens so this can't happen.
if [ "${IS_HPT_BASELINE:-0}" = "1" ]; then
  ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE_HPT:-2}"
  ROLLOUT_N="${ROLLOUT_N:-4}"
  VERIFY_N="${VERIFY_N:-4}"
  # PIVOT to 'soft' (2026-06-03): the paper's adaptive 'switch' router trains
  # well (score 0.11->0.49) but is fatally unstable on this 4xH200/verl-fork at
  # bsz=128 — five runs each died/hung in update_actor at a different step
  # (1/16/31/71) via an abrupt worker death ("code 2"), and a post-crash Ray
  # teardown can hang (defeating checkpoint+resume). 'soft' is UNIFORM (every
  # prompt: n on-policy + 1 off-policy SFT, no per-rank data imbalance) and
  # completed 200 steps cleanly before (bsz=32 -> 0.435, so it is leak-free);
  # its bounded-token micro-batches (dynamic_bsz + max_token below) keep peak
  # memory equal to switch's, and at the full bsz=128 (4x the data of the 0.435
  # run) it has a real shot at clearing sft. Set HPT_UNIFY_STRATEGY=switch to
  # revert if the fork is ever fixed.
  HPT_UNIFY_STRATEGY="${HPT_UNIFY_STRATEGY:-soft}"
  # Use the EXACT proven sft/grpo runtime (those completed cleanly): the global
  # defaults give use_dynamic_bsz=False, ppo_micro_batch_size=2, gpu_mem=0.40,
  # max_response=4096. The first bsz=128 soft run instead used dynamic_bsz=True +
  # gpu_mem=0.30 + max_token=6144 (hacks I'd added for switch's memory crashes)
  # and DIVERGED to MATH-500=0.101 (job of 2026-06-04): a model-wrecking
  # over-imitation, two suspects vs the working bsz=32 soft (0.435) — dynamic_bsz
  # mis-scaling the custom SFT loss, and coef=1.0 giving 4x SFT exposure at
  # bsz=128. Fix both: revert to dynamic_bsz=False (matches sft/grpo, no custom
  # loss scaling) and set sft_loss_coef=0.3 (the reference's explicit
  # Qwen2.5-Math-1.5B value, exp_scripts/train.sh:28) to curb over-SFT. soft is
  # uniform + leak-free so dynamic_bsz=False/micro=2 fits and completes.
  USE_DYNAMIC_BSZ="False"
  SFT_LOSS_COEF="0.3"
  # save_freq for the retry loop's resume; soft is expected to finish in one go.
  SAVE_FREQ=10
else
  ROLLOUT_N="${ROLLOUT_N:-4}"
  VERIFY_N="${VERIFY_N:-4}"
fi
export HPT_UNIFY_STRATEGY
VAL_N="${VAL_N:-16}"
ROLLOUT_MAX_PREFIX_LEN="${ROLLOUT_MAX_PREFIX_LEN:-8192}"
SAVE_FREQ="${SAVE_FREQ:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

mkdir -p "${RUN_ROOT}"
rm -rf "${SPLIT_DIR}" "${CHECKPOINT_DIR}" "${RUN_ROOT}"/eval_* "${FINAL_METRICS}" "${TRAIN_LOG}"
mkdir -p "${SPLIT_DIR}" "${CHECKPOINT_DIR}"
echo "Run profile: run_name=${RUN_NAME} is_hpt_baseline=${IS_HPT_BASELINE} train_batch_size=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N} verify_n=${VERIFY_N} ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE}" | tee -a "${TRAIN_LOG}"

if [ ! -f "${MODEL_RUN_DIR}/config.json" ]; then
  rm -rf "${MODEL_RUN_DIR}"
  mkdir -p "${MODEL_RUN_DIR}"
  cp -as "${MODEL_DIR}/." "${MODEL_RUN_DIR}/"
fi
if [ -L "${MODEL_RUN_DIR}/config.json" ]; then
  rm -f "${MODEL_RUN_DIR}/config.json"
  cp "${MODEL_DIR}/config.json" "${MODEL_RUN_DIR}/config.json"
fi

python - <<PY
import json
from pathlib import Path

config_path = Path("${MODEL_RUN_DIR}") / "config.json"
with config_path.open() as f:
    cfg = json.load(f)

cfg["max_position_embeddings"] = 16384
cfg["sliding_window"] = 16384
cfg["rope_theta"] = 40000

with config_path.open("w") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
    f.write("\\n")
PY

if [ ! -f "${PKG_DATA_DIR}/AIME24/test.parquet" ] || [ ! -f "${PKG_DATA_DIR}/AMC23/test.parquet" ] || [ ! -f "${PKG_DATA_DIR}/MATH-500/test.parquet" ]; then
  (
    cd "${PKG_DATA_DIR}"
    python preprocess.py
  )
fi

python "${TASK_DIR}/scripts/make_shared_splits.py" \
  --train-input "${OPENR1_PATH}" \
  --aime24-input "${EVAL_AIME24_PATH}" \
  --amc23-input "${EVAL_AMC23_PATH}" \
  --math500-input "${EVAL_MATH500_PATH}" \
  --out-dir "${SPLIT_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --train-source-spec "${TRAIN_SOURCE_SPEC}" \
  --train-source-selection "${TRAIN_SOURCE_SELECTION}" \
  --aime24-size "${AIME24_SIZE}" \
  --amc23-size "${AMC23_SIZE}" \
  --math500-size "${MATH500_SIZE}" \
  --internal-aime24-size "${INTERNAL_AIME24_SIZE}" \
  --internal-amc23-size "${INTERNAL_AMC23_SIZE}" \
  --internal-math500-size "${INTERNAL_MATH500_SIZE}" \
  --seed "${SEED_VALUE}"

python - <<PY
train_size = int("${TRAIN_SIZE}")
train_batch_size = int("${TRAIN_BATCH_SIZE}")
total_training_steps = int("${TOTAL_TRAINING_STEPS}")
verify_n = int("${VERIFY_N}")
actual_updates = max(total_training_steps - 1, 1)

prompt_passes = train_batch_size * actual_updates / train_size
verify_exposure = prompt_passes * verify_n

print(
    "Derived training coverage: "
    f"optimizer_updates={actual_updates} "
    f"prompt_passes_per_sample={prompt_passes:.3f} "
    f"verify_exposure_per_sample={verify_exposure:.3f}",
    flush=True,
)
PY

cd "${PKG_DIR}/hpt/verl"

# The 'switch' off-policy path has a gradual resource leak that abruptly kills a
# worker after a variable number of steps (runs died at step 31, then 71 once we
# added GPU headroom — the crash point shifts under the SAME seed, so it is
# accumulation over steps, not a bad batch or a fixed memory wall). Headroom
# alone can't reach 200 steps. Fix: save_freq>0 + resume_mode=auto + this in-job
# retry loop. On each crash we restart the python process (the leak resets) and
# verl auto-resumes from the latest checkpoint in CHECKPOINT_DIR, so a single
# SLURM allocation completes all 200 steps across a few restarts — no re-queue,
# no lost place in line. The initial wipe (above, before the loop) gives a clean
# attempt 1; the loop never wipes, so attempts 2+ resume.
MAX_TRAIN_ATTEMPTS="${MAX_TRAIN_ATTEMPTS:-10}"
TRAIN_RC=1
for attempt in $(seq 1 "${MAX_TRAIN_ATTEMPTS}"); do
  echo "===== TRAIN ATTEMPT ${attempt}/${MAX_TRAIN_ATTEMPTS} (resume_mode=auto from ${CHECKPOINT_DIR}) =====" | tee -a "${TRAIN_LOG}"
  ray stop --force >/dev/null 2>&1 || true
  sleep 5
python -m verl.mix_src.main_mix_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.grpo_use_std=False \
  data.train_files="${SPLIT_DIR}/mixed_train.parquet" \
  data.val_files="[\"${SPLIT_DIR}/AIME24_eval.parquet\",\"${SPLIT_DIR}/AMC23_eval.parquet\",\"${SPLIT_DIR}/MATH-500_eval.parquet\"]" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.path="${MODEL_RUN_DIR}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.offline_loss_type=sft \
  actor_rollout_ref.actor.off_policy_normalize=False \
  actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
  actor_rollout_ref.actor.sft_loss_coef="${SFT_LOSS_COEF}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_remove_token_mean=True \
  actor_rollout_ref.actor.loss_remove_clip=True \
  actor_rollout_ref.ref.use_ref=False \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${LOG_PROB_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.n_verify="${VERIFY_N}" \
  actor_rollout_ref.rollout.n_val="${VAL_N}" \
  actor_rollout_ref.rollout.val_temperature=0.6 \
  +actor_rollout_ref.rollout.val_top_p=0.95 \
  actor_rollout_ref.rollout.max_prefix_len="${ROLLOUT_MAX_PREFIX_LEN}" \
  trainer.logger=[console] \
  trainer.project_name=mlsbench_hpt \
  trainer.experiment_name="shared_seed${SEED_VALUE}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.unify_strategy="${HPT_UNIFY_STRATEGY:-switch}" \
  trainer.switch_gate=0 \
  trainer.switch_gate_off=0 \
  trainer.remove_sfted_data=False \
  trainer.max_optim_to_keep=1 \
  trainer.resume_mode=auto \
  trainer.default_hdfs_dir=null \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  +trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  data.reward_impl_version=6 \
  2>&1 | tee -a "${TRAIN_LOG}"
  TRAIN_RC=${PIPESTATUS[0]}
  # Success = clean exit AND a final validation line in the log. The metric-line
  # check is the real success signal because Hydra/Ray can mask a crash as exit 0.
  if [ "${TRAIN_RC}" -eq 0 ] && grep -q "val/test_score/MATH-500" "${TRAIN_LOG}"; then
    echo "Training completed cleanly on attempt ${attempt}." | tee -a "${TRAIN_LOG}"
    break
  fi
  echo "Attempt ${attempt} exited rc=${TRAIN_RC} without final metrics; restarting with resume." | tee -a "${TRAIN_LOG}"
  sleep 10
done
# Fail if no attempt produced final validation metrics, so the orchestrator
# records a failure instead of launching the metric relay on an empty output.
if ! grep -q "val/test_score/MATH-500" "${TRAIN_LOG}"; then
  echo "ERROR: training did not complete after ${MAX_TRAIN_ATTEMPTS} attempts." >&2
  exit 1
fi
TRAIN_RC=0
TRAIN_LOG_PATH="${TRAIN_LOG}" python - <<'PY' | tee "${FINAL_METRICS}" | tee -a "${TRAIN_LOG}"
import os
import re
from pathlib import Path

log_path = Path(os.environ["TRAIN_LOG_PATH"])
text = log_path.read_text()

metric_dict = None

for line in reversed(text.splitlines()):
    if "step:" not in line or "val/test_score/" not in line:
        continue
    pairs = {
        key: float(value)
        for key, value in re.findall(
            r"(val/test_score/(?:AIME24|AMC23|MATH-500)):([0-9.]+)",
            line,
        )
    }
    if len(pairs) == 3:
        metric_dict = pairs
        break

if metric_dict is None:
    extracted = {}
    for short_key in ("AIME24", "AMC23", "MATH-500"):
        values = re.findall(
            rf"val/test_score/{re.escape(short_key)}['\"]?:\s*([0-9.]+)",
            text,
        )
        if values:
            extracted[f"val/test_score/{short_key}"] = float(values[-1])
    if len(extracted) == 3:
        metric_dict = extracted

if metric_dict is None:
    raise RuntimeError(f"Could not find final validation metrics in {log_path}")

print(f"Final validation metrics: {metric_dict}")
PY

# Guard: if no valid final metrics were extracted (e.g. training crashed before
# completing — Hydra/Ray can swallow the trainer error and exit 0), fail here so
# the orchestrator records a failure instead of launching the ~2.5h metric relay
# that would wait on a final_metrics file that never becomes valid.
if ! grep -q "Final validation metrics" "${FINAL_METRICS}" 2>/dev/null; then
  echo "ERROR: no valid final validation metrics in ${FINAL_METRICS}; treating training as failed." >&2
  exit 1
fi

#!/usr/bin/env bash
# cc_rl_multisource.sh -- GRPO RL on the MIXED parquet (frontiersmith_synth +
# frontiercs_research + mlsbench_rl in one file), 9B.
#
# MIXING MECHANISM (zero verl changes): rollout is always the AgentLoopManager;
# routing is PER ROW via the parquet's agent_name column:
#   * single_turn_agent  -> synth + research rows (built-in agent); their reward
#     comes from the streaming RewardLoopWorker -> default_compute_score, which
#     dispatches on data_source (sandbox harness / evaluator subprocess).
#   * mlsbench_agent     -> MLS rows (registered via the agent_loop_config_path
#     override below); reward computed in-loop -> AgentLoopOutput.reward_score.
# All scores land per-row in rm_scores. FS_PERTASK_REWARD_NORM=1 rescales the
# single-turn 0-100 scorers to [0,1], matching the MLS task-score scale.
#
# Data: python scripts/prepare_multisource_rl_parquet.py [--smoke|--balance]
# Offline routing check: scripts/check_multisource_reward_routing.py
#
# Usage (full run):   sbatch slurm/cc_rl_multisource.sh
# Usage (1-step smoke, all sources in the single batch):
#   sbatch --time=06:00:00 --export=ALL,EXPERIMENT_NAME=ms_smoke1,TOTAL_TRAINING_STEPS=1,\
#     TRAIN_DATA=$PWD/data/multisource_rl/train_smoke.parquet,ROLLOUT_N=2,SAVE_FREQ=100000 \
#     slurm/cc_rl_multisource.sh
#
#SBATCH --job-name=rl-multisource
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"
mkdir -p logs

# ---- node guards (carried from cc_rl_frontiersmith_synth.sh) ----
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load cudatoolkit/12.8 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
[ -x "$CUDA_HOME/bin/nvcc" ] && export PATH="$CUDA_HOME/bin:$PATH"
unset XDG_RUNTIME_DIR || true
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# MLS episodes run up to MLS_RL_EPISODE_TIMEOUT (3000s) inside rollout; keep the
# NCCL heartbeat above the longest single reward/episode stall.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# vLLM penalty fast path (2026-08-13). presence_penalty=1.5 (our eval protocol)
# puts vLLM's sampler on a CPU path that rebuilds a [batch, max_len] int64
# tensor EVERY decode step: measured 162 ms/step at B=128 L=32768, ~70% of the
# step, with GPUs idling at 0-37%. scripts/vllm_penalty_fastpath.py appends only
# the new tokens (bit-identical output, 37x faster). Inert unless this is 1;
# install/revert with scripts/apply_vllm_penalty_fastpath.sh.
export FS_VLLM_PENALTY_FASTPATH="${FS_VLLM_PENALTY_FASTPATH:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Host-side stack limit (2026-08-12). Our login shell has `ulimit -s unlimited`,
# which Slurm propagates into the job; glibc then hands spawned threads a small
# FIXED stack instead of a growable one, and OpenBLAS's per-thread buffers
# overflow it -> SIGSEGV in numpy called from any worker thread. MLS host-side
# result parsing does exactly that (dgp.truth() -> np.linalg.solve inside a
# ThreadPoolExecutor), which kills the whole `mlsbench agent` process; 25 of the
# 84 training tasks are causal-* and hit that path. Verified by an independent
# repro: crashes with ONE worker thread and on 8 cores, i.e. it is the stack
# limit, NOT LAPACK concurrency (our earlier OPENBLAS_NUM_THREADS=1 theory was
# wrong, and that workaround needlessly serialised host-side parsing).
ulimit -s 8192 2>/dev/null || true

# ---- research reward env (same as cc_rl_frontiersmith_synth.sh) ----
export FRONTIERCS_RESEARCH_CPU_TIMEOUT="${FRONTIERCS_RESEARCH_CPU_TIMEOUT:-300}"
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_depot}"
export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_env}"
export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-64}"

# ---- synth reward env ----
export FRONTIERSMITH_SYNTH_ROOT="${FRONTIERSMITH_SYNTH_ROOT:-$PROJECT_ROOT/../innovation_prior/frontiersmith_synth}"
export FRONTIERSMITH_SYNTH_FAIL_SOFT="${FRONTIERSMITH_SYNTH_FAIL_SOFT:-1}"
export FRONTIERSMITH_SYNTH_MAX_WALL="${FRONTIERSMITH_SYNTH_MAX_WALL:-300}"
# Host-RAM guards (2026-08-11, post rlv3 OOM forensics). Scoring is streaming
# (per-sample as each rollout finishes); concurrency = 8 reward workers x
# MAX_CONC. With the 8G per-child RLIMIT below, 4x8=32 concurrent scorings is
# a safe worst case (32x8G=256G) and removes the scoring serialization tail.
export FRONTIERSMITH_SYNTH_MAX_CONC="${FRONTIERSMITH_SYNTH_MAX_CONC:-4}"
export FSX_CHILD_MEM_MB="${FSX_CHILD_MEM_MB:-8192}"
# MLS episode subtree cap (the actual rlv3 OOM killer: model-generated bench
# code at 140G+/process). Inherited by every test/evaluate child.
export MLS_RL_EPISODE_MEM_MB="${MLS_RL_EPISODE_MEM_MB:-16384}"

# ---- per-task reward norm: single-turn 0-100 -> [0,1] to match MLS scale ----
export FS_PERTASK_REWARD_NORM="${FS_PERTASK_REWARD_NORM:-1}"
export FS_PERTASK_REWARD_NORM_TARGET="${FS_PERTASK_REWARD_NORM_TARGET:-1.0}"

# ======================================================================
# ANTI-DEAD-GROUP KNOBS. Both default OFF -> this block is a no-op and the
# run behaves exactly as it did before these were added.
#
# Measured problem: 17.3% of all 5,120 GRPO groups across the 4-arm campaign
# (885 groups = 14,160 rollouts) had zero score spread, so their advantage was
# exactly 0 and they emitted NO gradient. Those dead groups are truncation-
# driven: 85.5% truncated vs 32.4% in live groups at base step 10, and
# P(dead | zero of 16 samples completed) = 0.93, with median length pinned at
# exactly 32,768 and slope +-0.0 tokens/step over 20 steps.
#
# NOTE: these are plain env vars, not hydra overrides. They reach the Ray
# actors the same way FS_PERTASK_REWARD_NORM already does (driver env ->
# raylet -> workers).
# ======================================================================

# ---- Feature 1: per-prompt adaptive resampling ("group deepening") ----
# When a prompt's group comes back flat, draw MORE samples of THAT SAME prompt
# (n -> 2n -> 4n ... up to ADAPTIVE_N_MAX) instead of swapping in a fresh prompt
# the way DAPO's filter_groups does. The enlarged group supplies the GRPO
# baseline; exactly ROLLOUT_N rows are folded back into the update carrying
# Horvitz-Thompson weights, so ppo_mini_batch_size, the token-mean denominator
# and seq-balancing are all untouched. See verl/trainer/ppo/adaptive_sampling.py.
export ADAPTIVE_N_ENABLE="${ADAPTIVE_N_ENABLE:-0}"
# v2 trigger (2026-08-10): no_positive = deepen while the group has NO raw-
# positive sample (judged PRE-penalty), drop the group from the loss entirely
# (mask out of the token-mean denominator) if the budget ends without one.
# zero_variance = the v1 behaviour (deepen flat groups, never drop). v1's
# trigger goes blind once the overlong penalty adds variance to all-failed
# groups -- and those penalty-only groups uptrain wrong-but-short answers.
export ADAPTIVE_N_TRIGGER="${ADAPTIVE_N_TRIGGER:-no_positive}"
export ADAPTIVE_N_POS_EPS="${ADAPTIVE_N_POS_EPS:-0.0}"  # raw > this = positive
export ADAPTIVE_N_MAX="${ADAPTIVE_N_MAX:-128}"          # ceiling on the deepened group size
export ADAPTIVE_N_GROWTH="${ADAPTIVE_N_GROWTH:-2.0}"    # 16 -> 32 -> 64 -> 128
export ADAPTIVE_N_EPS="${ADAPTIVE_N_EPS:-1e-6}"         # zero_variance: max-min <= eps counts as flat
export ADAPTIVE_N_KEEP="${ADAPTIVE_N_KEEP:-subsample}"  # subsample (safe) | all (ragged batch!)
export ADAPTIVE_N_STRATIFY="${ADAPTIVE_N_STRATIFY:-1}"  # protect the rare success from the draw
export ADAPTIVE_N_MAX_PROMPTS="${ADAPTIVE_N_MAX_PROMPTS:-0}"  # 0 = deepen every flat group
# Budget guard. One extra mlsbench_agent sample is a whole multi-minute episode,
# not one generate call, so an unbounded deepening round can dominate step time.
# Finite by default (review M2): unlimited extra rollouts + a requeue flood can
# spike one step by ~1.8k extra 32k-token generations. 512 caps the worst case
# at ~half a normal step's generation volume. 0 = unlimited (opt-in only).
export ADAPTIVE_N_MAX_EXTRA="${ADAPTIVE_N_MAX_EXTRA:-512}"
# In-wave deepening (v2.5): the extra rollouts are launched by the agent-loop
# workers, so the trainer-side ADAPTIVE_N_MAX_EXTRA no longer governs them --
# each worker enforces its own share. Codex review 2026-08-12 finding 3: export
# it explicitly (default = the global budget split over the worker count) so the
# cap is never silently unlimited.
export ADAPTIVE_N_INWAVE="${ADAPTIVE_N_INWAVE:-0}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-2}"
export ADAPTIVE_N_MAX_EXTRA_PER_WORKER="${ADAPTIVE_N_MAX_EXTRA_PER_WORKER:-$(( ADAPTIVE_N_MAX_EXTRA / AGENT_NUM_WORKERS ))}"
export ADAPTIVE_N_AGENTS="${ADAPTIVE_N_AGENTS:-}"             # "" = all; e.g. single_turn_agent
export ADAPTIVE_N_SEED="${ADAPTIVE_N_SEED:-0}"
# v2.1 (2026-08-10): (A) clamp = no row with non-positive RAW reward may get
# positive advantage (closes the penalty-mean<0 hole by construction);
# (B) requeue = first-time budget-exhausted prompts retry ONCE in a later
# batch before being dropped for good (single-epoch loader would otherwise
# lose them forever); count injected per step is rounded down to a multiple
# of PPO_MINI_BATCH_SIZE so the mini-batch split never goes ragged.
export ADAPTIVE_CLAMP_ZERO_RAW="${ADAPTIVE_CLAMP_ZERO_RAW:-1}"
export ADAPTIVE_REQUEUE_ENABLE="${ADAPTIVE_REQUEUE_ENABLE:-1}"
# v2.4 (2026-08-12, user directive): pipelined deepening. The synchronous ladder
# fires an extra generate_sequences AFTER the main wave drains (measured 124
# min/step of low-occupancy GPU). With overlap=1 no extra wave is issued: a
# starving group is masked out of the step and its prompt returns in the NEXT
# main wave at 2x size, so deepening tokens are generated at full width.
# DEFAULT OFF (2026-08-12): Codex xhigh review found 7 defects in the overlap
# path, incl. a stale group_sizes that made the ladder cycle at 32 forever. With
# ADAPTIVE_N_MAX=32 the synchronous path only runs ONE deepening round, so
# overlap now buys ~18 min/step -- not worth shipping unreviewed machinery.
# Findings 1 and 5 are fixed in ray_trainer.py; 2/3/4 are not. Do not enable
# without a cross-step integration test (16 fail -> 32 fail -> ceiling drop).
export ADAPTIVE_N_OVERLAP="${ADAPTIVE_N_OVERLAP:-0}"
export ADAPTIVE_REQUEUE_AFTER_STEPS="${ADAPTIVE_REQUEUE_AFTER_STEPS:-1}"
export ADAPTIVE_REQUEUE_MAX_FRAC="${ADAPTIVE_REQUEUE_MAX_FRAC:-0.25}"

# ---- Feature 2: DAPO soft overlong punishment ----
# Linear ramp over the last FS_OVERLONG_BUFFER_LEN tokens of the response
# budget, reaching -FS_OVERLONG_PENALTY_FACTOR exactly at MAX_RESPONSE_LENGTH.
# Applied in AgentLoopWorker._postprocess because that is the only point BOTH
# agents share -- mlsbench_agent scores never reach a reward manager at all, so
# neither shipped DAPORewardManager can do this job (see the file header of
# verl/utils/reward_score/overlong_penalty.py for the file:line argument).
# SCALE: with FS_PERTASK_REWARD_NORM=1 rewards live in [0,1], so factor=1.0
# spans twice the reward range. 0.5-1.0 is the sensible band.
# DAPO overlong FILTERING (the other half of DAPO; we shipped only the soft
# punishment). Truncated rollouts are removed from the loss entirely instead of
# being trained on with a full-magnitude negative reward. Measured motivation:
# they held 56-60% of the gradient in the two rlv10 arms that destabilised, and
# that gradient cannot teach stopping (no stop action exists in a truncated
# trajectory), so entropy rose and truncation fed itself.
# Kept OFF by default: filtering DID stop the entropy blowup, but it also made
# over-length free (truncated samples become invisible to the loss) and the fixed
# arm drifted 7600 -> 19764 tokens / 5% -> 27% truncation by s11. seq-mean below
# already removes the 7.6x weight advantage those samples had, which was the
# actual disease; the penalty then supplies the length signal at a sane scale.
export FS_OVERLONG_FILTER="${FS_OVERLONG_FILTER:-0}"
# token-mean weights every token equally, so a 32768-token runaway carries ~7.6x
# the gradient of a healthy 4300-token rollout. seq-mean-token-mean averages
# within a sequence first, removing that length amplification.
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
export FS_OVERLONG_PENALTY="${FS_OVERLONG_PENALTY:-0}"
export FS_OVERLONG_BUFFER_LEN="${FS_OVERLONG_BUFFER_LEN:-4096}"
# 2026-08-14: measured on rlv10 step 1 -- at factor 0.5 the mean penalty (-0.287)
# was 2.9x the mean correctness signal (+0.101), i.e. the reward was mostly a
# length function; 81% of base's score gain came from the penalty shrinking, not
# from solving more. 0.15 puts the penalty at 0.86x correctness.
export FS_OVERLONG_PENALTY_FACTOR="${FS_OVERLONG_PENALTY_FACTOR:-0.15}"
export FS_OVERLONG_LOG="${FS_OVERLONG_LOG:-1}"
# FS_OVERLONG_MAX_RESP_LEN defaults to data.max_response_length; override only
# if you deliberately want a different reference length.
export FS_OVERLONG_MAX_RESP_LEN="${FS_OVERLONG_MAX_RESP_LEN:-}"

# ---- MLS-Bench RL episode env (read by mlsbench_agent_loop.py in Ray workers) ----
export MLS_RL_MLSBENCH_ROOT="${MLS_RL_MLSBENCH_ROOT:-/scratch/gpfs/CHIJ/bohan/MLS-Bench-train}"
export MLS_RL_DATA_ROOT="${MLS_RL_DATA_ROOT:-/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data}"
export MLS_RL_WORKER_PYTHON="${MLS_RL_WORKER_PYTHON:-/home/bl3615/miniconda3/bin/python}"
export MLS_RL_WORKER_SCRIPT="${MLS_RL_WORKER_SCRIPT:-$PROJECT_ROOT/scripts/mlsbench_rl_episode_worker.py}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ms_qwen35_9b_grpo}"
export MLS_RL_OUTPUT_BASE="${MLS_RL_OUTPUT_BASE:-$PROJECT_ROOT/outputs/mls_rl/${EXPERIMENT_NAME}}"
export MLS_RL_MAX_STEPS="${MLS_RL_MAX_STEPS:-8}"       # parquet extra_info.budget overrides per row
export MLS_RL_MAX_TESTS="${MLS_RL_MAX_TESTS:-1}"
export MLS_RL_USE_REPLACE="${MLS_RL_USE_REPLACE:-1}"
export MLS_RL_MAX_CONC_TESTS="${MLS_RL_MAX_CONC_TESTS:-3}"
export MLS_RL_EPISODE_TIMEOUT="${MLS_RL_EPISODE_TIMEOUT:-3000}"
export MLS_RL_NUDGE_LIMIT="${MLS_RL_NUDGE_LIMIT:-2}"
export MLS_RL_KEEP_WORKSPACE="${MLS_RL_KEEP_WORKSPACE:-0}"
export MLSBENCH_SCHEDULER_MANAGED=1
export MLSBENCH_NO_PREBUILT=1
mkdir -p "$MLS_RL_OUTPUT_BASE"

# ---- data: mixed parquet (scripts/prepare_multisource_rl_parquet.py) ----
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/multisource_rl/train.parquet}"
export VAL_DATA="${VAL_DATA:-$TRAIN_DATA}"
export ALEBENCH_VAL_DATA="${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/__none__.parquet}"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "[multisource] ERROR: $TRAIN_DATA missing. Build it with:" >&2
    echo "  .venv-vllm023/bin/python scripts/prepare_multisource_rl_parquet.py --smoke" >&2
    exit 1
fi

# ---- training schedule ----
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-60}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-100000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export PROJECT_NAME="${PROJECT_NAME:-rl_multisource}"
export CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/rl_multisource/${EXPERIMENT_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-$PROJECT_ROOT/outputs/rl_multisource_rollout/${EXPERIMENT_NAME}}"

# ---- GRPO shape (MLS episodes are minutes-long; keep the batch modest) ----
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"   # user 2026-08-04: scale up from July 32x8
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"  # 64/16 = 4 updates per rollout batch (user); each sees 16x16=256 sequences
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export VAL_N="${VAL_N:-1}"

# ---- memory/speed (proven 9B 4-GPU recipe) ----
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"  # 9B FSDP-sharded over 8 GPUs leaves plenty; more KV cache = higher rollout concurrency = faster steps

# ---- context: MLS full84 prompts reach 25,966 tokens -> 28,672 prompt budget.
# Packing arithmetic (verl asserts MAXTOKLEN x ULYSSES_SP >= prompt+response):
# 30720 x 2 = 61440 = 28672 prompt + 32768 response. Generation floor 32K is a
# standing user rule; SP=2 keeps per-GPU activation load at the proven 9B level.
# Rollout sampling MUST match the July-proven synth run (job rlfsx_q35_inst_start,
# 30% nonzero at step 1, FCS 7.05->11.03). Stock verl defaults are pure sampling
# (top_p 1.0 / top_k -1 / presence 0) which made Qwen3.5 ramble to the 32k cap
# without closing </think> -> all-zero rewards in the ms_smoke3 run. These three
# values are also the eval protocol, so training optimizes the evaluated behavior.
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
export ROLLOUT_PRESENCE_PENALTY="${ROLLOUT_PRESENCE_PENALTY:-1.5}"
export ROLLOUT_MIN_P="${ROLLOUT_MIN_P:-0.0}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-26624}"   # measured MLS max 25952 (+672 slack); synth max is only 2257
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"  # user rule: generation NEVER below 32K
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-59392}"   # = 26624 + 32768, the union worst case (MLS prompt + full generation budget)
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-43008}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-29696}"  # x ULYSSES_SP(2) = 59392; per-GPU load BELOW the proven July 9B run (45056/GPU at SP=1)
export ULYSSES_SP="${ULYSSES_SP:-2}"
# Dynamic batching pairs with ULYSSES_SP per the runner's own guidance: micro-batches
# are packed to a token budget instead of a fixed 1 sequence, so the very uneven
# lengths here (synth generates ~32k, MLS prompts ~26k) stop wasting padding.
# torch.compile must go OFF with it: varying micro-batch shapes trigger guard
# failures/recompiles that hung a straggler rank at the step-1 checkpoint barrier.
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"

export NGPU="${NGPU:-4}"
export TP="${TP:-1}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.5-9B}"

mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

echo "[multisource] TRAIN_DATA=$TRAIN_DATA"
echo "[multisource] MODEL=$MODEL_PATH NGPU=$NGPU steps=$TOTAL_TRAINING_STEPS batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N}"
echo "[multisource] SYNTH_ROOT=$FRONTIERSMITH_SYNTH_ROOT  MLS_ROOT=$MLS_RL_MLSBENCH_ROOT"
echo "[multisource] ctx: prompt=$MAX_PROMPT_LENGTH resp=$MAX_RESPONSE_LENGTH model_len=$MAX_MODEL_LEN"
echo "[multisource] MLS episodes: max_steps=$MLS_RL_MAX_STEPS max_tests=$MLS_RL_MAX_TESTS timeout=${MLS_RL_EPISODE_TIMEOUT}s -> $MLS_RL_OUTPUT_BASE"
echo "[multisource] adaptive-n: enable=$ADAPTIVE_N_ENABLE trigger=$ADAPTIVE_N_TRIGGER max=$ADAPTIVE_N_MAX growth=$ADAPTIVE_N_GROWTH keep=$ADAPTIVE_N_KEEP stratify=$ADAPTIVE_N_STRATIFY agents='${ADAPTIVE_N_AGENTS:-ALL}' max_extra=$ADAPTIVE_N_MAX_EXTRA overlap=$ADAPTIVE_N_OVERLAP inwave=$ADAPTIVE_N_INWAVE(per-worker $ADAPTIVE_N_MAX_EXTRA_PER_WORKER)"
echo "[multisource] overlong:   enable=$FS_OVERLONG_PENALTY buffer=$FS_OVERLONG_BUFFER_LEN factor=$FS_OVERLONG_PENALTY_FACTOR"

# ---- window guard (same as synth launcher) ----
if [ "${FRESH_START:-0}" != "1" ] && [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  _done=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "$_done" -ge "$TOTAL_TRAINING_STEPS" ] 2>/dev/null; then
    echo "[window-guard] latest ckpt step=$_done >= $TOTAL_TRAINING_STEPS -> nothing to do."
    exit 0
  fi
fi

# ---- sandbox preflight (synth rows FAIL_SOFT to silent 0.0 without a sandbox;
# see cc_rl_frontiersmith_synth.sh 2026-07-30 note). Always run: the mixed
# parquet always contains synth rows. ----
if [ "${SKIP_BWRAP_PREFLIGHT:-0}" != "1" ]; then
  _ISORUN="$FRONTIERSMITH_SYNTH_ROOT/harness/isorun.py"
  if bwrap --ro-bind / / true 2>/dev/null; then
    echo "[preflight] sandbox backend: bwrap"
  elif ISORUN_BACKEND=apptainer python3 "$_ISORUN" 2>/dev/null | grep -q "sandbox=apptainer"; then
    echo "[preflight] bwrap dead; apptainer sandbox verified via real isorun self_check"
    export ISORUN_BACKEND=apptainer
  else
    echo "FATAL: no candidate sandbox on $(hostname): bwrap AND apptainer both dead." >&2
    echo "       synth reward would FAIL_SOFT to 0.0 for EVERY synth rollout." >&2
    exit 1
  fi
fi

# ---- run the proven GRPO runner + the MLS agent-loop registration.
# agent_loop_config_path adds mlsbench_agent to the registry; single_turn_agent
# is built-in, so synth/research rows are untouched. num_workers bounds the
# per-node Apptainer fan-out (workers x MLS_RL_MAX_CONC_TESTS).
# Run a job-PRIVATE copy of the runner. bash reads a script incrementally by byte
# offset, so editing the shared file mid-run makes an already-running job resume at
# a shifted offset and execute garbage -- that is exactly how job 12064549 died
# ("line 326: trainer.save_freq=5: command not found") when the runner was patched
# while it ran. The snapshot also records what each job actually executed.
# NOTE: the snapshot MUST sit exactly one level under PROJECT_ROOT -- the runner
# derives PROJECT_ROOT as "$(dirname $0)/..", so a deeper path (e.g. logs/x/y.sh)
# makes it resolve to logs/ and the venv lookup fails (job 12074024).
RUNNER_SNAPSHOT="${RUNNER_SNAPSHOT:-$PROJECT_ROOT/.runner_snapshots/run_verl_grpo_${SLURM_JOB_ID:-$$}.sh}"
mkdir -p "$(dirname "$RUNNER_SNAPSHOT")"
cp scripts/run_verl_grpo_frontiercs_qwen35_9b.sh "$RUNNER_SNAPSHOT"
echo "[multisource] runner snapshot: $RUNNER_SNAPSHOT"

exec bash "$RUNNER_SNAPSHOT" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/config/mlsbench_agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers="${MLS_RL_AGENT_WORKERS:-2}" \
    "$@"

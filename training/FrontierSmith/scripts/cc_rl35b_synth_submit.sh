#!/usr/bin/env bash
# Submit GRPO RL on frontiersmith_synth for the Qwen3.6-35B-A3B (MoE) line.
# Reuses the PROVEN 9B chain (slurm/cc_rl_frontiersmith_synth.sh ->
# scripts/run_verl_grpo_frontiercs_qwen35_9b.sh) with 35B-specific memory knobs.
#
# 35B-A3B on 8xH200 (141G) memory plan (2026-07-14 feasibility analysis, verl
# fork weight-sync path traced 2026-07-14):
#   - verl FSDP trainer keeps fp32 params: 35.95B*4B/8 = 16.7 G/GPU (+16.7 G grads).
#   - Adam fp32 m+v = 268 G total (33.5 G/GPU) -> MUST be offloaded
#     (ACTOR_OPTIMIZER_OFFLOAD=True): resident would repeat the 9B save-OOM.
#     CPU side fits easily (node 1.5 T, job asks GPUS*120G).
#   - rollout: vLLM sleeps at level 2 during update_actor (weights DESTROYED,
#     re-synced each step via CUDA-IPC buckets) -> GMU only sizes the AWAKE pool.
#     TP=1 -> full 67 G bf16 weight copy per GPU inside the pool: GMU 0.65
#     (=91.6 G) = 67 G weights + ~24 G KV; hybrid GDN arch has only 10/40
#     full-attention layers so KV is small. Awake-phase total = pool 91.6 +
#     fp32 shard 16.7 (+grads) ~= 128 G < 141 G.
#   - weight sync: fused 3D expert tensors (experts.gate_up_proj [256,2*512,2048]
#     = 2.0 GiB fp32 max) fit the 8192 MB IPC bucket; vLLM 0.23 qwen3_5 loader
#     natively maps fused HF names -> w13/w2 (verified in code). MTP keys are
#     skipped on both sides.
#   - update_actor: micro_batch=1 + gradient checkpointing + expandable_segments.
#   - OOM backoff order: GMU 0.65->0.55; MAXRESP 32768->24576; TP=1->2 (GMU 0.35).
#   - actor attn = sdpa; GDN layers run the torch fallback (flash-linear-attention
#     not in .venv-vllm023) -- slower, same as every 9B qwen3_5 run.
#   - transformers 5.12.1 loads the LF-saved UNFUSED expert checkpoints
#     (mlp.experts.N.gate_proj.weight) into the FUSED live module
#     (mlp.experts.gate_up_proj) via the qwen2_moe conversion mapping --
#     round-trip verified exact on 2026-07-14.
#
# Usage:
#   bash scripts/cc_rl35b_synth_submit.sh                     # smoke: 3 steps, 8x4
#   STEPS=20 TB=32 RN=8 MB=16 SMOKE=0 ONLY=r32s01 bash scripts/cc_rl35b_synth_submit.sh
set -euo pipefail

ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
SCRIPT="$FS/slurm/cc_rl_frontiersmith_synth.sh"
# TRAIN/VAL default to frontiersmith_synth; override RL_TRAIN_DATA/RL_VAL_DATA to run
# a different RL data source (e.g. data/frontiercs/research.parquet, data_source
# frontiercs_research -> the research reward adapter). The reward routes by the
# parquet's own data_source field, so no extra wiring is needed.
TRAIN="${RL_TRAIN_DATA:-$FS/data/frontiersmith_synth/train.parquet}"
VAL="${RL_VAL_DATA:-$FS/data/frontiersmith_synth/full.parquet}"
SYNTH_ROOT="$ROOT/innovation_prior/frontiersmith_synth"

GPUS="${GPUS:-8}"
TP="${TP:-1}"
CPUS=$(( GPUS * 8 ))     # ailab cap: 8 cores/GPU
# HOST-RAM lesson (smoke 11169862, 2026-07-14): 8-GPU 35B run OOM-KILLED at 960G
# host mem during ref_log_prob -- fp32 Adam offload (268G) + ref fp32 params on
# CPU (140G) + Ray object store (~30% of cgroup) + 8 workers' host overhead
# exceed GPUS*120. The full node (1.5T) is ours anyway with 8 GPUs -> take it.
MEMG="${MEM:-$(( GPUS >= 8 ? 1450 : GPUS * 120 ))}"

SMOKE="${SMOKE:-1}"
if [ "$SMOKE" = "1" ]; then
  STEPS="${STEPS:-3}"; SAVE="${SAVE:-2}"
  TB="${TB:-8}"; RN="${RN:-4}"; MB="${MB:-8}"
  WALL="${WALL:-08:00:00}"
else
  STEPS="${STEPS:-20}"; SAVE="${SAVE:-5}"
  TB="${TB:-32}"; RN="${RN:-8}"; MB="${MB:-16}"
  WALL="${WALL:-23:59:00}"
fi

# Length config: response cap RAISED 32768->40960 (2026-07-20 fix; still >=32K, the
# user's hard floor). The 32768 run truncated 52% of step-1 rollouts, and truncated/
# rambling rollouts skew longer==worse -> length collapse. 40960 lets long-correct
# finish. model_len 45056 unchanged (fits 40960 resp + <=1617 prompt = 42577).
MAXRESP="${MAXRESP:-40960}"
MAXLEN="${MAXLEN:-45056}"
# vLLM share: TP=1 needs the full 67G weight copy inside the GMU budget.
GMU="${GMU:-0.65}"
# 35B memory knobs (see header): params on GPU, optimizer offloaded to CPU.
APO="${APO:-False}"      # ACTOR_PARAM_OFFLOAD
AOO="${AOO:-True}"       # ACTOR_OPTIMIZER_OFFLOAD
RPO="${RPO:-True}"       # REF_PARAM_OFFLOAD
# LONG-SEQ MEMORY (2026-07-15): 35B params/GPU shard leaves less room than 9B, so a
# single 32768-tok sequence's backward activation OOMs (step2, GPU5, 3GB short).
# Fix = Ulysses sequence parallelism: split each long seq across SP GPUs => per-GPU
# token load = PPO_MAX_TOKEN_LEN_PER_GPU (NOT the full 34816). Requires dynamic bsz.
# Keeps RESP=32768. per-GPU cap 24576 << 34816 that OOM'd => ~30GB headroom.
# WINNER (uly6 smoke 11276936 + formal 20-step both arms): SP=4 / MAXTOKLEN=12288.
# SP=2/24576 was TRIED (11206885/11215908) and still OOM'd at step-2 backward AND
# broke the packing assertion.
#
# PACKING ASSERT (2026-07-27, killed 6 jobs pre-step-1 — read this before touching
# lengths). verl asserts  MAXTOKLEN*SP >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH.
# The comment here used to compare against MAX_MODEL_LEN (45056) — WRONG: MAX_MODEL_LEN
# is only vLLM's rollout cap. With the default prompt 10240 + resp 40960 the real
# max_seq_len is 51200 > 12288*4 = 49152 => AssertionError in compute_log_prob before
# step 1. (The 12288 "winner" was validated at SP=6: 12288*6 = 73728, so it never hit
# this; pairing it with the SP=4 OOM fix created a config that was never valid.)
# FIX = shrink the prompt budget, NOT raise MAXTOKLEN: the longest real prompt is 2835
# tokens (synth 1165 max 2257, research 64 max 2835), so 10240 was 3.6x oversized.
# MAXPROMPT=4096 => max_seq_len 45056 <= 49152 with 4k slack, AND lowers peak activation
# (guards the step-2 OOM that killed MAXTOKLEN=13312). RESP stays 40960 (>=32K rule).
#
# UPDATE-ACTOR OOM (2026-07-28). With MAXPROMPT=4096 the packing assert passes, but the
# run then died at `actor_rollout_update_actor` with "Triton Error [CUDA]: out of memory"
# ~5h in (step 2) — the SAME wall MAXTOKLEN=13312 hit. Lesson: cutting MAXPROMPT fixed the
# ASSERT but did NOT cut memory. Under dynamic bsz the per-GPU activation load is the
# packing bin = MAXTOKLEN, independent of how long any individual sequence is. 12288 was
# still 12288.
# REAL FIX = raise SP so the SAME sequence is split over more GPUs, which lets MAXTOKLEN
# drop proportionally: SP 4->8 with MAXTOKLEN 12288->6144 keeps MAXTOKLEN*SP = 49152 >=
# 45056 (assert still satisfied) while HALVING the per-GPU token load, which is the thing
# that OOMs. Qwen3.6-35B-A3B has num_attention_heads=16, so SP=8 gives 2 query heads/rank
# (kv heads 2 are repeated) — divisibility is fine. Costs some SP all-to-all traffic.
DYNBSZ="${DYNBSZ:-True}"         # USE_DYNAMIC_BSZ (token-budget micro-batching)
ULYSP="${ULYSP:-8}"             # ULYSSES_SP (shard each seq across all 8 GPUs)
MAXTOKLEN="${MAXTOKLEN:-6144}"  # PPO_MAX_TOKEN_LEN_PER_GPU (per-GPU token cap; halved with SP)
MAXPROMPT="${MAXPROMPT:-4096}"  # MAX_PROMPT_LENGTH (real max is 2835 tokens)

# Preflight: reproduce verl's packing assert here so a bad length combo fails in 1s at
# submit time instead of burning a node for ~75min and dying before step 1.
_bin=$(( MAXTOKLEN * ULYSP )); _seq=$(( MAXPROMPT + ${MAXRESP:-40960} ))
if [ "$_bin" -lt "$_seq" ]; then
  echo "FATAL: MAXTOKLEN*ULYSP = $MAXTOKLEN*$ULYSP = $_bin < MAX_PROMPT+MAX_RESP = $MAXPROMPT+${MAXRESP:-40960} = $_seq" >&2
  echo "       verl asserts max_token_len >= max_seq_len. Lower MAXPROMPT (real max prompt is 2835 tok)" >&2
  echo "       or raise MAXTOKLEN -- but MAXTOKLEN=13312 OOM'd at step-2 backward, so prefer lowering MAXPROMPT." >&2
  exit 1
fi
# SAVE-TIME HOST-OOM FIX (2026-07-16, smoke 11266326): the step-2 checkpoint save's
# offload_to_cpu spike stacked on optim(288G)+ref offloads OOM-killed a worker at
# host RAM (node 1464G phys, we ask 1450G => ~14G margin). Cap Ray's plasma store
# (default ~435G on this node; rollout data is only ~1G) to free ~385G for the save.
RAYOBJ="${RAYOBJ:-50000000000}" # RAY_OBJECT_STORE_BYTES (50G)
# torch.compile OFF for 35B: dynamic-bsz varying shapes + torch.compile hang a
# straggler rank at the step-1 checkpoint barrier (smoke 11266326/11270526 hung in
# save after Inductor compile warnings; sharded save never wrote). Static-shape 9B
# keeps it ON. See run_verl_grpo_frontiercs_qwen35_9b.sh header.
TORCHCOMPILE="${TORCHCOMPILE:-False}"  # USE_TORCH_COMPILE
# SAVE-TIME HOST-OOM REAL FIX (2026-07-16, smoke 11273365 proved cap+compile-off
# both OOM at save): verl's sharded save hardcodes offload_to_cpu=True, materializing
# each rank's shard on host on top of optim(288G)+ref offloads => cgroup OOM (MaxRSS
# 145G/worker × 8). vLLM sleeps at save time so the GPU has room to serialize the
# shard in place. VERL_CKPT_OFFLOAD_TO_CPU=0 (patched into fsdp_checkpoint_manager.py)
# keeps shards on GPU => torch.save streams per-tensor, tiny host peak.
CKPTOFFLOAD="${CKPTOFFLOAD:-0}"  # VERL_CKPT_OFFLOAD_TO_CPU (0 => save from GPU)
# Eval-matched rollout sampling (same as the 9B synth arm).
RTOPP="${RTOPP:-0.95}"; RTOPK="${RTOPK:-20}"; RPP="${RPP:-1.5}"
KEEP="${KEEP:-10}"
# ---- ANTI-COLLAPSE (DAPO) knobs, 2026-07-20 ----
# The first 35B synth RL (vanilla GRPO) death-spiralled: reward inverted-U (peak
# s5 0.23 -> s20 0.08), response len 26k->9k, all-zero groups 0%->31%, entropy
# 0.77->0.27. Defaults below are the FIX (this submitter is 35B-only; vanilla is
# proven broken here). Override to A/B against the old behavior.
#   clip-higher: let low-prob exploratory tokens rise -> resists entropy collapse.
#   MAXRESP raise (32768->40960, still >=32K): step-1 had 52% truncation at 32768;
#     letting long-correct rollouts finish weakens the "longer==worse" advantage that
#     drives length collapse. Memory-safe: per-GPU load stays MAXTOKLEN-bounded, and
#     MAXTOKLEN*SP=49152 >= MAXPROMPT+MAXRESP=45056 (this inequality is the packing
#     assert -- it only holds because MAXPROMPT was cut to 4096; see the block above).
#   filter_groups: NOT wired in main_ppo (no-op) -> kept False; densify later via
#     ROLLOUT_N or offline difficulty filter if late gradient-starvation persists.
CLIP_HIGH="${CLIP_HIGH:-0.28}"                 # was 0.2 (symmetric); DAPO 0.28
CLIP_LOW="${CLIP_LOW:-0.2}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"   # no-op in main_ppo; see note
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_final_reward}"
FILTER_GROUPS_MAX_GEN_BATCHES="${FILTER_GROUPS_MAX_GEN_BATCHES:-8}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"          # keep tiny (was already 0.001)
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"            # clip-higher is the entropy control

declare -A MODELS=(
  [r32s01]="$ROOT/models_sft/lora_q36_35bA3b_clean_nom_r32_s01_merged"
  [base]="$ROOT/models/Qwen3.6-35B-A3B"
  [r32wd03s01]="$ROOT/models_sft/lora_q36_35bA3b_clean_nom_r32_wd03_s01_merged"
)
ORDER=(r32s01)
[ -n "${ONLY:-}" ] && ORDER=($ONLY)

cd "$FS"
for tag in "${ORDER[@]}"; do
  mp="${MODELS[$tag]}"
  [ -f "$mp/config.json" ] || { echo "ABORT: model missing for $tag: $mp" >&2; exit 1; }
  for f in preprocessor_config.json processor_config.json video_preprocessor_config.json chat_template.jinja; do
    [ -f "$mp/$f" ] || echo "WARN: $mp/$f missing (vLLM/AutoProcessor may fail)" >&2
  done
  exp="rl35b_${tag}${EXPSUFFIX:-}$([ "$SMOKE" = "1" ] && echo _smoke || true)"
  ck="$FS/checkpoints/rl_frontiersmith_synth/$exp"
  ro="$FS/outputs/rl_frontiersmith_synth_rollout/$exp"
  COMMON="MODEL_PATH=$mp,TRAIN_DATA=$TRAIN,VAL_DATA=$VAL,EXPERIMENT_NAME=$exp,PROJECT_NAME=rl_frontiersmith_synth,CKPT_DIR=$ck,ROLLOUT_DIR=$ro,FRONTIERSMITH_SYNTH_ROOT=$SYNTH_ROOT,FRONTIERSMITH_SYNTH_FAIL_SOFT=1,TOTAL_TRAINING_STEPS=$STEPS,SAVE_FREQ=$SAVE,TEST_FREQ=100000,NGPU=$GPUS,TP=$TP,MAX_PROMPT_LENGTH=$MAXPROMPT,MAX_RESPONSE_LENGTH=$MAXRESP,MAX_MODEL_LEN=$MAXLEN,MAX_NUM_BATCHED_TOKENS=$MAXLEN,TRAIN_BATCH_SIZE=$TB,PPO_MINI_BATCH_SIZE=$MB,ROLLOUT_N=$RN,MAX_ACTOR_CKPT_TO_KEEP=$KEEP,GPU_MEMORY_UTILIZATION=$GMU,ROLLOUT_TOP_P=$RTOPP,ROLLOUT_TOP_K=$RTOPK,ROLLOUT_PRESENCE_PENALTY=$RPP,ACTOR_PARAM_OFFLOAD=$APO,ACTOR_OPTIMIZER_OFFLOAD=$AOO,REF_PARAM_OFFLOAD=$RPO,USE_DYNAMIC_BSZ=$DYNBSZ,ULYSSES_SP=$ULYSP,PPO_MAX_TOKEN_LEN_PER_GPU=$MAXTOKLEN,RAY_OBJECT_STORE_BYTES=$RAYOBJ,USE_TORCH_COMPILE=$TORCHCOMPILE,VERL_CKPT_OFFLOAD_TO_CPU=$CKPTOFFLOAD,CLIP_LOW=$CLIP_LOW,CLIP_HIGH=$CLIP_HIGH,KL_LOSS_COEF=$KL_LOSS_COEF,ENTROPY_COEFF=$ENTROPY_COEFF,FILTER_GROUPS_ENABLE=$FILTER_GROUPS_ENABLE,FILTER_GROUPS_METRIC=$FILTER_GROUPS_METRIC,FILTER_GROUPS_MAX_GEN_BATCHES=$FILTER_GROUPS_MAX_GEN_BATCHES"
  jid=$(sbatch --parsable \
    ${DEP:+--dependency=$DEP} \
    --job-name="$exp" --time="$WALL" \
    --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="${MEMG}G" \
    --export=ALL,$COMMON,FRESH_START="${FRESH_START:-1}" \
    "$SCRIPT")
  extra=""
  # Formal runs get continuation windows (like the 9B chain): afterany resumes
  # with FRESH_START=0; the launcher's window-guard makes them no-ops once
  # TOTAL_TRAINING_STEPS is reached.
  if [ "$SMOKE" != "1" ]; then
    w2=$(sbatch --parsable --dependency=afterany:$jid \
      --job-name="${exp}_w2" --time="$WALL" \
      --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="${MEMG}G" \
      --export=ALL,$COMMON,FRESH_START=0 \
      "$SCRIPT")
    w3=$(sbatch --parsable --dependency=afterany:$w2 \
      --job-name="${exp}_w3" --time="$WALL" \
      --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="${MEMG}G" \
      --export=ALL,$COMMON,FRESH_START=0 \
      "$SCRIPT")
    extra=" w2=$w2 w3=$w3"
  fi
  echo "submitted $exp -> $jid$extra (model=$mp, ${GPUS}GPU TP=$TP, ${TB}x${RN} mini=$MB, resp=$MAXRESP, GMU=$GMU, dynbsz=$DYNBSZ sp=$ULYSP maxtok/gpu=$MAXTOKLEN, steps=$STEPS)"
done

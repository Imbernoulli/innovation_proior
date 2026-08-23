#!/usr/bin/env bash
# =============================================================================
# Research-GPU (21 Triton problems) DEDICATED SCORING JOB -- 1 GPU, serial.
#
# Rescores EXISTING research generations (samples.jsonl with --save-text) for
# the 21 GPU problems only, using the official evaluator.py on a whole free
# GPU with the research_overlay env. Reported as a SEPARATE "Research-GPU"
# column -- never merged into the Research-64 history.
#
# Why a dedicated job: the self-hosted eval pool (cc_eval_pool_selfhosted.sh)
# serves vLLM at GPU_MEMORY_UTILIZATION=0.90 x TP=2 and does not export
# FRONTIERCS_RESEARCH_PYTHON, so under it the 21 GPU evaluators return broken
# uniform zeros (qknorm additionally errors on missing flashinfer). Generation
# is fine; only GPU-problem scoring must move to a job like this one.
#
# Usage:
#   sbatch --job-name=res-gpu-<TAG> \
#     --export=ALL,TAG=<arm>,SAMPLES="<samples.jsonl or output dir> [more]" \
#     slurm/cc_eval_research_gpu_score.sh
#   # ailab instead of pli: add  --partition=ailab --account=ailab --qos=<...>
#   # (sbatch CLI flags override the #SBATCH defaults below)
#
# Optional env: OUT_DIR, PROBLEMS (comma list), N_SAMPLES (default 5),
#   FRONTIERCS_RESEARCH_GPU_TIMEOUT (default 2400 = campaign research budget),
#   SKIP_CONTROL=1, RESUME=1 (default on).
#
# Budget: measured on A100, 21x5 base-arm samples score in ~0.5h; correct
# (benchmark-running) solutions are slower -- 3h wall is comfortable.
# =============================================================================
#SBATCH --job-name=cc-research-gpu-score
#SBATCH --partition=pli
#SBATCH --account=goedelprover
#SBATCH --qos=pli-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/research_gpu

: "${TAG:?set TAG (arm name, e.g. anchor_base)}"
: "${SAMPLES:?set SAMPLES (space-separated samples.jsonl files / eval output dirs)}"

# Evaluator interpreter: research_overlay = sft_lf (torch 2.10 + triton 3.6)
# + flashinfer 0.6.12 (qknorm) + the CPU-subset deps. GPU problems need
# nothing beyond the base + flashinfer; no network at runtime.
_OVL=/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python
[ -x "$_OVL" ] || _OVL=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf/bin/python
export FRONTIERCS_RESEARCH_PYTHON="${FRONTIERCS_RESEARCH_PYTHON:-$_OVL}"
# 2400s per evaluator run: the SAME budget the campaign gives research
# evaluators (FRONTIERCS_RESEARCH_CPU_TIMEOUT / REQUEST_TIMEOUT are 2400).
export FRONTIERCS_RESEARCH_GPU_TIMEOUT="${FRONTIERCS_RESEARCH_GPU_TIMEOUT:-2400}"
# RLIMIT_AS guard: 24 GiB (the RL default) kills large-VA GPU evaluators
# (vector_addition/2_28 dies rc=127; scores ~50 at >=48 GiB). 64 GiB here.
export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-64}"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/tmp}"
# Triton kernel cache: keep it node-local so concurrent arms don't clobber.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${SLURM_JOB_ID:-$$}}"

# GPU keep-alive against the 90-min 0%-util auto-cancel: some samples spend
# minutes in CPU-side triton compilation before touching the GPU.
"$FRONTIERCS_RESEARCH_PYTHON" -c "import torch,time
while True:
    (torch.randn(256,256,device='cuda')@torch.randn(256,256,device='cuda')).sum().item(); time.sleep(30)" >/dev/null 2>&1 &
KEEPALIVE_PID=$!
trap 'kill $KEEPALIVE_PID >/dev/null 2>&1 || true' EXIT INT TERM

ARGS=(--tag "$TAG" --n-samples "${N_SAMPLES:-5}"
      --timeout "$FRONTIERCS_RESEARCH_GPU_TIMEOUT")
[ -n "${OUT_DIR:-}" ]   && ARGS+=(--out-dir "$OUT_DIR")
[ -n "${PROBLEMS:-}" ]  && ARGS+=(--problems "$PROBLEMS")
[ "${SKIP_CONTROL:-0}" = "1" ] && ARGS+=(--skip-control)
[ "${RESUME:-1}" = "1" ] && ARGS+=(--resume)

# SAMPLES is intentionally word-split (space-separated list).
# shellcheck disable=SC2086
"$PROJECT_ROOT/.venv/bin/python" scripts/frontiercs_research_gpu_rescore.py \
  --samples $SAMPLES "${ARGS[@]}"

echo "[res-gpu] DONE tag=$TAG"

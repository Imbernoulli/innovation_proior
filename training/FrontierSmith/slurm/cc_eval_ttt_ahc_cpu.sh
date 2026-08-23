#!/usr/bin/env bash
# EVALUATION recipe (TTT-Discover, native offline scorer): compile + run a C++
# AtCoder-Heuristic-Contest (AHC) solution against TTT-Discover's CACHED public
# test inputs and report the ALE-Bench-style absolute score. CPU-only, no LLM,
# no Tinker, no internet -> fully offline.
#
# NOTE on scope: TTT-Discover's *model* loop (ttt_discover.discover) is hard-gated
# to GPT-OSS-20b/120b and runs RL through the Tinker CLOUD API (TINKER_API_KEY +
# internet). It therefore CANNOT serve/evaluate the local Qwen3.5-9B on ailab.
# This script evaluates a *candidate solution* (released or model-produced .cpp)
# with TTT-Discover's own judge, which is the only TTT-Discover scoring path that
# works offline. To score Qwen3.5-9B as a discovery model offline, use the
# OpenEvolve-based recipe: slurm/cc_eval_ttt_discover_openevolve_ailab.sh
#
# Usage (TAG names the run; CODE selects which .cpp to judge):
#   sbatch slurm/cc_eval_ttt_ahc_cpu.sh <TAG> [PROBLEM] [CODE_CPP]
#   PROBLEM in {ahc039, ahc058, both} (default both)
#   CODE_CPP default = TTT-Discover/results/algorithm-design/<problem>.cpp (released)
# Example:
#   sbatch slurm/cc_eval_ttt_ahc_cpu.sh released both
#
# Output: TTT-Discover/outputs/cc_eval_ttt_ahc_<TAG>/summary.json
# Metric: per-problem ALE-Bench "absolute_score" (HIGHER is better; AHC raw score),
#         summarized as total_absolute_score / mean_absolute_score over the 150
#         cached cases, plus num_accepted/num_cases. Read total_absolute_score.

#SBATCH --job-name=cc-eval-ttt-ahc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail

TTT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover
FS_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

TAG="${1:-${TAG:-released}}"
PROBLEM="${2:-${PROBLEM:-both}}"

mkdir -p "$FS_ROOT/logs"
cd "$TTT_ROOT"

OUT="$TTT_ROOT/outputs/cc_eval_ttt_ahc_${TAG}"
mkdir -p "$OUT" "$OUT/logs"

# gcc-toolset/14 provides g++ new enough to compile the AHC C++ solutions.
module load gcc-toolset/14 2>/dev/null || echo "WARN: could not load gcc-toolset/14; relying on system g++" >&2

# Offline env (no model download/datasets needed here, but keep it consistent).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$TTT_ROOT:${PYTHONPATH:-}"
export ALE_BENCH_CACHE="$TTT_ROOT/examples/ahc/lib/cache"
export ALE_BENCH_LOG_DIR="$OUT/logs"
export ALE_BENCH_STRIP_GMP_LINK=1

# Activate a Python env that has ray (the FrontierSmith vLLM venv has it).
VENV_DIR="${VENV_DIR:-$FS_ROOT/.venv-vllm023}"
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

python scripts/evaluate_ahc_released.py \
  --repo "$TTT_ROOT" \
  --problem "$PROBLEM" \
  --num-cpus "${SLURM_CPUS_PER_TASK:-32}" \
  --out "$OUT/summary.json"

echo "=== DONE. Read the final number from: $OUT/summary.json (total_absolute_score per problem) ==="
cat "$OUT/summary.json" || true

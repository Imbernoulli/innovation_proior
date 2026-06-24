#!/usr/bin/env bash
# EVALUATION recipe (TTT-Discover MATH task, offline, with a LOCAL model):
# score Qwen3.5-9B as a *discovery* model on a TTT-Discover mathematics task
# (autocorrelation inequalities / Erdos overlap) by driving the SAME objective
# through a local vLLM server + the OpenEvolve test-time search loop. Thinking ON.
#
# WHY this wrapper exists: TTT-Discover's native loop (ttt_discover.discover) is
# hard-asserted to GPT-OSS-20b/120b and trains via the Tinker CLOUD API, which
# needs TINKER_API_KEY + internet. ailab compute nodes have NO internet and the
# target model is Qwen3.5-9B, so the native TTT-Discover loop cannot run here.
# The TTT-Discover math objectives are mathematically identical to ThetaEvolve's
# autocorrelation OpenEvolve evaluators (verified: same convolution-based score
# in TTT-Discover/scripts/verify_released_math_results.py and the OpenEvolve
# evaluator_modular.py), so this gives an apples-to-apples, OFFLINE way to score
# the local model on the TTT-Discover math objective.
#
# TASK -> TTT-Discover math objective mapping:
#   second_autocorr_inequality  ==  TTT-Discover "AC2"  (C2 lower bound, HIGHER better)
#   third_autocorr_inequality   ==  TTT-Discover "AC3"
#   (first_autocorr_inequality / erdos have no local-vLLM smoke config checked in)
#
# Usage:
#   sbatch slurm/cc_eval_ttt_discover_openevolve_ailab.sh <MODEL_DIR> <TAG> [TASK] [ITERATIONS]
# Example (base instruct, AC2):
#   sbatch slurm/cc_eval_ttt_discover_openevolve_ailab.sh \
#     /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B base_instruct
#
# This delegates to the ThetaEvolve OpenEvolve eval wrapper with a TTT-math TASK
# default, so the serving / thinking / offline / scoring logic is shared.

set -euo pipefail
FS_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

MODEL_DIR="${1:-${MODEL_DIR:-$FS_ROOT/models/Qwen3.5-9B}}"
TAG="${2:-${TAG:-base_instruct}}"
TASK="${3:-${TASK:-second_autocorr_inequality}}"   # = TTT-Discover AC2
ITERATIONS="${4:-${ITERATIONS:-12}}"

# Re-tag so outputs land under a TTT-named directory but reuse the shared engine.
exec sbatch \
  --job-name="cc-eval-ttt-math-${TAG}" \
  "$FS_ROOT/slurm/cc_eval_theta_openevolve_ailab.sh" \
  "$MODEL_DIR" "ttt_${TAG}" "$TASK" "$ITERATIONS"

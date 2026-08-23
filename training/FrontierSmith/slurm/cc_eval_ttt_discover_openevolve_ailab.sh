#!/usr/bin/env bash
# EVALUATION recipe (TTT-Discover MATH task, offline, with a LOCAL model):
# score Qwen3.5-9B as a *discovery* model on a TTT-Discover mathematics task
# (autocorrelation inequalities) by driving the SAME objective through a local
# vLLM server + the OpenEvolve test-time search loop. Thinking ON.
#
# ============================ FAITHFULNESS NOTE ============================
# This is an ENGINE PROXY, not the native TTT-Discover protocol. It faithfully
# reproduces the SCORING METRIC but substitutes a DIFFERENT search engine:
#   * Native TTT-Discover (ttt_discover.discover -> ttt_discover/rl/train.py):
#       genuine TEST-TIME REINFORCEMENT LEARNING. Each step it samples a group
#       of programs (group_size=64, groups_per_batch=8) from the CURRENT LoRA
#       policy, scores them with the reward evaluator, computes entropic
#       adaptive-beta advantages + KL-to-base penalty, and UPDATES THE MODEL
#       WEIGHTS via the Tinker cloud training API (LoRA rank 32, lr 4e-5,
#       importance-sampling loss, up to num_epochs=50 steps). The model itself
#       learns on the single problem at hand -- that is the "TTT" in TTT-Discover.
#   * This proxy (OpenEvolve / ThetaEvolve): the model weights are FROZEN. A
#       MAP-Elites island evolutionary loop mutates programs in-context for
#       ITERATIONS steps. No gradient step, no weight update, no RL.
# Both maximize the SAME objective, so the raw best-objective number is directly
# comparable to the released TTT-Discover math metric, but the OPTIMIZER is not
# the paper's test-time-training loop. Read this as "frozen-model in-context
# search on the TTT-Discover objective", not "TTT-Discover reproduced".
#
# WHY the native loop is unavailable here (exact blocker chain):
#   1) `import ttt_discover` fails at import: needs `chz` then `tinker`
#      (pyproject hard-deps chz>=0.3 + tinker>=0.7), neither installed in the
#      offline vLLM venv. VERIFIED: ModuleNotFoundError: No module named 'chz'.
#   2) Even installed, discover_impl() ASSERTS model_name in
#      {gpt-oss-120b, gpt-oss-20b} (ttt_discover/discovery.py:74) -> refuses Qwen.
#   3) train.py uses tinker.ServiceClient(base_url=None) +
#      create_lora_training_client_async -> the Tinker CLOUD training service
#      (needs TINKER_API_KEY + internet). ailab compute nodes have NO internet.
#   Net: the native weight-updating TTT loop CANNOT run on ailab as-is. To make
#   it faithful you must vendor a local Tinker-compatible trainer (verl/LF) and
#   drop the gpt-oss assert -- see slurm/cc_run_ttt_discover_native_stub.sh.
#
# SCORING is faithful (VERIFIED byte-for-byte):
#   The OpenEvolve compute_objective_value() for AC1/AC2 is line-for-line the
#   same math as the native evaluate_sequence_ac1/ac2 (examples/ac_inequalities/
#   env.py) AND the released-result checker (scripts/verify_released_math_results.py,
#   which reproduces the paper's AC2=0.95918 / AC1=1.50286 / Erdos=0.380875).
#   The parser's headline `best_combined_score` == raw objective_value for these
#   MAXIMIZE tasks (VanillaObjectiveScoring passes it through), so the headline
#   IS the native metric. The config `score_transform` only rescales the SEPARATE
#   `rl_normalized_reward` field and must NOT be read as the metric.
#
# TASK -> TTT-Discover math objective mapping (CORRECTED):
#   first_autocorr_inequality   == TTT-Discover "AC1" (C1 upper bound, LOWER better;
#                                  proxy metric = R = 2n*max(f*f)/sum(f)^2)
#   second_autocorr_inequality  == TTT-Discover "AC2" (C2 lower bound, HIGHER better)  [DEFAULT]
#   erdos                        == TTT-Discover Erdos min-overlap (LOWER better)
#                                  -- NO OpenEvolve proxy example exists; native-only.
#   third_autocorr_inequality   == *** NOT A TTT-DISCOVER TASK ***. TTT-Discover has
#                                  ONLY AC1 + AC2 (see README key-results + native
#                                  ac_inequalities/env.py, which raises on anything
#                                  but 'ac1'/'ac2'). "C3" is a ThetaEvolve/OpenEvolve
#                                  -only objective. Runs with TASK=third_autocorr_*
#                                  DO NOT measure any TTT-Discover metric.
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
# Default = AC2, the canonical TTT-Discover math task with a faithful proxy.
TASK="${3:-${TASK:-second_autocorr_inequality}}"   # = TTT-Discover AC2 (HIGHER better)
ITERATIONS="${4:-${ITERATIONS:-12}}"

# Faithfulness guard: only AC1 / AC2 correspond to real TTT-Discover metrics.
case "$TASK" in
  first_autocorr_inequality|second_autocorr_inequality) : ;;  # faithful AC1/AC2 proxies
  third_autocorr_inequality)
    echo "WARNING: TASK=third_autocorr_inequality is NOT a TTT-Discover task." >&2
    echo "         TTT-Discover math = AC1 + AC2 + Erdos ONLY (no 'AC3'/'C3')." >&2
    echo "         This run measures a ThetaEvolve-only objective, not a TTT metric." >&2
    echo "         Set TTT_ALLOW_NON_TTT_TASK=1 to proceed anyway." >&2
    [ "${TTT_ALLOW_NON_TTT_TASK:-0}" = "1" ] || { echo "Refusing. Use second_autocorr_inequality for a faithful AC2 eval." >&2; exit 2; }
    ;;
  erdos|erdos_min_overlap)
    echo "ERROR: Erdos min-overlap has NO OpenEvolve proxy example in ThetaEvolve." >&2
    echo "       It is native-TTT-only. Use the native runner (see" >&2
    echo "       slurm/cc_run_ttt_discover_native_stub.sh) or score a released sequence" >&2
    echo "       with TTT-Discover/scripts/verify_released_math_results.py." >&2
    exit 2
    ;;
  *) echo "WARNING: unrecognized TASK=$TASK; proceeding but verify it has a qwen35_local smoke config." >&2 ;;
esac

# Re-tag so outputs land under a TTT-named directory but reuse the shared engine.
exec sbatch \
  --job-name="cc-eval-ttt-math-${TAG}" \
  "$FS_ROOT/slurm/cc_eval_theta_openevolve_ailab.sh" \
  "$MODEL_DIR" "ttt_${TAG}" "$TASK" "$ITERATIONS"

#!/usr/bin/env bash
# =============================================================================
# TTT-Discover NATIVE-LOOP runner  (FAITHFUL protocol) -- STUB + OFFLINE PATHS
# =============================================================================
# GOAL: run the ACTUAL ttt_discover.discover test-time-training loop, not the
# OpenEvolve frozen-model proxy. This script (a) runs every native path that
# WORKS offline today, and (b) documents the EXACT blocker + the concrete plan
# to make the weight-updating RL loop faithful on ailab.
#
# WHAT THE NATIVE LOOP ACTUALLY DOES (ttt_discover/rl/train.py):
#   For up to num_epochs (=50) steps on ONE problem:
#     1. Sample a GROUP of programs from the current LoRA policy
#        (group_size=64 rollouts x groups_per_batch=8), two-phase token budget
#        phase1_max_tokens=26000, temperature=1.0.
#     2. Execute each program in a CPU sandbox and score it with the task's
#        BaseRewardEvaluator (e.g. AC2 R(f), AHC absolute_score, Erdos C5).
#     3. Compute advantages = entropic_adaptive_beta (LOO), add KL-to-base
#        penalty (kl_penalty_coef=0.1), drop constant-reward groups.
#     4. forward_backward + optim_step (Adam, lr=4e-5, importance_sampling loss)
#        -> the MODEL WEIGHTS are updated. This gradient step IS "test-time
#        training"; the proxy has no analogue.
#     5. Save LoRA checkpoint every save_every=2 steps; report best sequence.
#   Model is HARD-GATED to openai/gpt-oss-{20b,120b} and training runs on the
#   Tinker CLOUD service (tinker.ServiceClient(base_url=None)).
#
# EXACT BLOCKER on ailab (all VERIFIED):
#   B1. import chz  -> ModuleNotFoundError (pyproject dep, not in offline venv).
#   B2. import tinker -> ModuleNotFoundError (pyproject dep tinker>=0.7).
#   B3. discovery.py:74 asserts model_name in {gpt-oss-120b, gpt-oss-20b}
#       -> refuses Qwen3.5-9B outright.
#   B4. train.py uses the Tinker CLOUD training API (needs TINKER_API_KEY +
#       outbound internet). ailab compute nodes have NO internet.
#   => The native WEIGHT-UPDATING loop cannot run here without a local
#      Tinker-compatible trainer. This is a HARD BLOCKER, not a config knob.
#
# PLAN to make it faithful (tracked; not yet implemented):
#   P1. Install chz + a LOCAL tinker shim, OR port train.py's 5 calls
#       (create_lora_training_client_async / forward_backward_async /
#        optim_step_async / save_weights_and_get_sampling_client_async /
#        compute_logprobs_async) onto the in-house verl/LlamaFactory LoRA
#        trainer we already run for RL (FrontierSmith/scripts/run_verl_grpo_*).
#   P2. Relax discovery.py:74 to allow Qwen3.5-9B (or run gpt-oss-20b locally
#       via vLLM if we want same-model faithfulness).
#   P3. Serve rollouts from the CURRENT LoRA (vLLM hot-reload / weight sync)
#       instead of a frozen server, so each step samples the updated policy.
#   P4. Reuse the native reward evaluators UNCHANGED (they are pure-numpy /
#       sandboxed and already offline-clean) so scoring stays byte-identical.
#   Until P1-P3 land, use cc_eval_ttt_discover_openevolve_ailab.sh as the
#   frozen-model proxy (faithful metric, non-native optimizer).
#
# WHAT THIS SCRIPT RUNS TODAY (offline, faithful SCORING, no Tinker):
#   MODE=verify_math : reproduce released AC1/AC2/Erdos numbers exactly.
#   MODE=ahc         : score a released/candidate AHC .cpp on the 150 cached
#                      public inputs with TTT-Discover's own ALE-Bench judge.
#   MODE=native      : attempt the real discover() loop; prints the blocker and
#                      exits non-zero unless TTT_FORCE_NATIVE=1 (for when P1-P3
#                      are done and tinker+chz are installed).
#
# Usage:
#   sbatch slurm/cc_run_ttt_discover_native_stub.sh verify_math
#   sbatch slurm/cc_run_ttt_discover_native_stub.sh ahc both
#   MODE=native sbatch slurm/cc_run_ttt_discover_native_stub.sh   # -> blocker report
#
#SBATCH --job-name=cc-ttt-native
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail

TTT_ROOT=/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover
FS_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

MODE="${1:-${MODE:-verify_math}}"
PROBLEM="${2:-${PROBLEM:-both}}"

mkdir -p "$FS_ROOT/logs"
cd "$TTT_ROOT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$TTT_ROOT:${PYTHONPATH:-}"

VENV_DIR="${VENV_DIR:-$FS_ROOT/.venv-vllm023}"
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

case "$MODE" in
  verify_math)
    echo "=== NATIVE released-math verification (faithful metric, no Tinker) ==="
    python scripts/verify_released_math_results.py
    echo "=== Compare against README key results: AC1=1.50287 AC2=0.9591 Erdos=0.380876 ==="
    ;;

  ahc)
    echo "=== NATIVE AHC scorer (TTT-Discover ALE-Bench judge, 150 cached inputs) ==="
    exec sbatch "$FS_ROOT/slurm/cc_eval_ttt_ahc_cpu.sh" "native_${PROBLEM}" "$PROBLEM"
    ;;

  native)
    echo "=== Attempting the NATIVE ttt_discover.discover loop ===" >&2
    if ! python -c "import chz, tinker" 2>/dev/null; then
      echo "BLOCKER B1/B2: chz and/or tinker not importable in this env." >&2
      python -c "import chz" 2>&1 | tail -1 >&2 || true
      python -c "import tinker" 2>&1 | tail -1 >&2 || true
    fi
    echo "BLOCKER B3: ttt_discover/discovery.py:74 asserts gpt-oss-{20b,120b}." >&2
    echo "BLOCKER B4: train.py uses the Tinker CLOUD service (needs TINKER_API_KEY + internet)." >&2
    if [ "${TTT_FORCE_NATIVE:-0}" != "1" ]; then
      echo "Refusing to launch the native weight-updating loop (blocked on this cluster)." >&2
      echo "See the header 'PLAN to make it faithful' (P1-P4). Set TTT_FORCE_NATIVE=1 once done." >&2
      exit 3
    fi
    # Only reached after P1-P3 are implemented and tinker+chz are installed.
    TASK="${TASK:-ac2}"
    echo "TTT_FORCE_NATIVE=1 -> launching discover_ac($TASK). Requires local trainer wiring." >&2
    python -c "from examples.ac_inequalities.env import discover_ac; discover_ac('${TASK}')"
    ;;

  *)
    echo "Unknown MODE=$MODE. Use verify_math | ahc | native." >&2
    exit 2
    ;;
esac

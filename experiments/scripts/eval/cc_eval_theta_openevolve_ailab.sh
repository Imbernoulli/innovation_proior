#!/usr/bin/env bash
# EVALUATION recipe: ThetaEvolve (no-RL, test-time in-context evolutionary search)
# of a HF model on an open optimization problem, via a LOCAL vLLM server + the
# OpenEvolve loop bundled in ThetaEvolve/openevolve_adapted. Thinking mode ON.
#
# This is the genuine "evaluate a model" path for ThetaEvolve: a fixed model is
# served by vLLM, OpenEvolve evolves a program for ITERATIONS steps, and we read
# the best program's score. Fully OFFLINE (model + repo are local).
#
# Usage (parameterized by MODEL_DIR + TAG, plus optional TASK/ITERATIONS):
#   sbatch slurm/cc_eval_theta_openevolve_ailab.sh <MODEL_DIR> <TAG> [TASK] [ITERATIONS]
# Examples:
#   sbatch slurm/cc_eval_theta_openevolve_ailab.sh \
#       /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/models/Qwen3.5-9B base_instruct
#   TASK=second_autocorr_inequality ITERATIONS=40 \
#       sbatch slurm/cc_eval_theta_openevolve_ailab.sh "$MODEL" mytag
#
# Output: ThetaEvolve/outputs/cc_eval_theta_<TAG>_<TASK>/job_<JOBID>/best/best_program_info.json
#         + a flat summary.json written next to it by parse_openevolve_best.py.
#
# Metric: OpenEvolve "combined_score" == task objective_value of the best evolved
#         program. Range/direction is task-specific (see config score_transform):
#           circle_packing_modular        higher better (~2.0-2.64; sum of radii)
#           second_autocorr_inequality    higher better (C2 lower bound, ~0.90-0.96)
#           third_autocorr_inequality     higher better
#         Read the final number from summary.json -> best_combined_score.

#SBATCH --job-name=cc-eval-theta
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail

FS_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
THETA_ROOT=/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve
OE_ROOT="$THETA_ROOT/openevolve_adapted"

# --- args (positional override env) ---
MODEL_DIR="${1:-${MODEL_DIR:-$FS_ROOT/models/Qwen3.5-9B}}"
TAG="${2:-${TAG:-base_instruct}}"
TASK="${3:-${TASK:-circle_packing_modular}}"
ITERATIONS="${4:-${ITERATIONS:-12}}"   # smoke default: 12 iters fits in <55min on 1 GPU
SEED="${SEED:-3407}"

if [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "ERROR: $MODEL_DIR has no config.json" >&2; exit 1
fi

mkdir -p "$FS_ROOT/logs"
cd "$THETA_ROOT"

# --- offline / HF env (ailab compute nodes have NO internet) ---
export PYTHONUNBUFFERED=1
# innovation time-conditioning system prompt at eval (matches SFT); year = the present
export EVAL_RESEARCHER_YEAR="${EVAL_RESEARCHER_YEAR:-2026}"
export HF_HOME="$FS_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
export OPENAI_API_KEY=EMPTY

# --- vLLM server (serves the model UNCHANGED; thinking is the Qwen3.5 template default) ---
export VLLM_PORT="${VLLM_PORT:-8021}"
export HOST=127.0.0.1
export MODEL_PATH="$MODEL_DIR"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export VLLM_CACHE_DIR="$FS_ROOT/.cache/vllm"

PORT="$VLLM_PORT" "$FS_ROOT/scripts/start_vllm_server.sh" &
VLLM_PID="$!"
cleanup() { kill "$VLLM_PID" >/dev/null 2>&1 || true; wait "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "Waiting for vLLM on :$VLLM_PORT ..."
for _ in $(seq 1 720); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then echo "vLLM server exited early" >&2; exit 1; fi
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null && echo "vLLM is up."

# --- OpenEvolve eval ---
cd "$OE_ROOT"
export PYTHONPATH="$OE_ROOT:$THETA_ROOT:${PYTHONPATH:-}"
export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"

INITIAL="examples/${TASK}/initial_programs/initial_program.py"
EVALUATOR="examples/${TASK}/evaluators/evaluator_modular.py"
CONFIG_SRC="examples/${TASK}/configs/config_${TASK}_qwen35_local_smoke.yaml"
if [ ! -f "$CONFIG_SRC" ]; then
  echo "ERROR: no local-vLLM smoke config for TASK=$TASK ($CONFIG_SRC)." >&2
  echo "Available: $(ls examples/${TASK}/configs/ 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

OUT="$THETA_ROOT/outputs/cc_eval_theta_${TAG}_${TASK}/job_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

# Patch api_base in a per-run config copy so VLLM_PORT is authoritative
# (the checked-in smoke configs hardcode an api_base port).
CONFIG="$OUT/config_used.yaml"
sed "s#api_base:.*#api_base: \"http://127.0.0.1:${VLLM_PORT}/v1\"#" "$CONFIG_SRC" > "$CONFIG"

# Record the initial-program (un-evolved) score for a before/after baseline.
# NOTE: $CONFIG is already an ABSOLUTE path ($OUT/config_used.yaml), so it must
# NOT be prefixed with $OE_ROOT/ — doing so produced a doubled, nonexistent path,
# silently fell back to the default (wrong-task) config, and recorded a bogus
# constant "Score: -0.2" baseline for every run. Use $CONFIG verbatim.
echo "=== initial program evaluator score (model-independent baseline) ===" >&2
OPENEVOLVE_CONFIG_PATH="$CONFIG" PYTHONPATH="$OE_ROOT:$THETA_ROOT" \
  python "$OE_ROOT/examples/${TASK}/evaluators/evaluator_modular.py" \
  "$OE_ROOT/$INITIAL" 2>&1 | tee "$OUT/initial_program_eval.txt" || \
  echo "initial eval returned nonzero" >&2

echo "=== OpenEvolve search: TASK=$TASK ITERATIONS=$ITERATIONS thinking=ON ===" >&2
python -m openevolve.cli "$INITIAL" "$EVALUATOR" \
  --config "$CONFIG" \
  --output "$OUT" \
  --iterations "$ITERATIONS" \
  --random-seed "$SEED" \
  --log-level INFO

# --- flatten the best score into summary.json ---
python "$FS_ROOT/scripts/parse_openevolve_best.py" \
  --run-dir "$OUT" --task "$TASK" --tag "$TAG" \
  --out "$OUT/summary.json"

echo "=== DONE. Read the final number from: $OUT/summary.json ==="
cat "$OUT/summary.json" || true

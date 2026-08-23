#!/usr/bin/env bash
# Standalone FrontierCS scoring diagnostic. Starts the offline judge
# (go-judge + node server via Apptainer) on a CPU node and re-scores a
# handful of known-good base-model samples directly with the official runner.
# CPU-only: compiling+running C++ is CPU work, no GPU needed.
#   sbatch slurm/diag_frontiercs_rescore_cpu.sh
#SBATCH --job-name=fs-fcs-diag
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export TMPDIR="/tmp"

# Unique ports + cgroup prefix per job to avoid collisions on co-located nodes.
export PORT="${PORT:-$(( 18000 + (SLURM_JOB_ID % 2000) ))}"
export GJ_PORT="${GJ_PORT:-$(( 25000 + (SLURM_JOB_ID % 2000) ))}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-4}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export GJ_CGROUP_PREFIX="gojudge-${SLURM_JOB_ID:-$$}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-diag-${SLURM_JOB_ID:-$$}}"

echo "PORT=$PORT GJ_PORT=$GJ_PORT cgroup=$GJ_CGROUP_PREFIX runtime=$RUNTIME_DIR"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "${JUDGE_PID}" >/dev/null 2>&1 || true
  wait "${JUDGE_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Wait for the judge HTTP API to come up.
READY=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.5
  if ! kill -0 "${JUDGE_PID}" >/dev/null 2>&1; then
    echo "judge launcher exited early" >&2
    exit 1
  fi
done
if [ "$READY" != 1 ]; then
  echo "Timed out waiting for judge at 127.0.0.1:${PORT}" >&2
  exit 1
fi
echo "judge health:"; curl -fsS "http://127.0.0.1:${PORT}/health"; echo

SAMPLES="${SAMPLES:-$PROJECT_ROOT/outputs/frontiercs_eval_qwen35_9b_official_vllm/samples.jsonl}"
OUT="${OUT:-$PROJECT_ROOT/outputs/frontiercs_eval_qwen35_9b_official_vllm/rescored_diag.jsonl}"

# First prove the judge produces nonzero on the SAME problems that scored
# nonzero in the working diag run (109,112,123), then on a broader sweep.
echo "=== Targeted: known-good problems 109,112,123 ==="
.venv/bin/python scripts/diag_frontiercs_rescore.py \
  --samples "$SAMPLES" \
  --judge-url "http://127.0.0.1:${PORT}" \
  --only-gt 109,112,123 \
  --limit 3

echo
echo "=== Broad sweep: first ${LIMIT:-16} distinct problems ==="
.venv/bin/python scripts/diag_frontiercs_rescore.py \
  --samples "$SAMPLES" \
  --judge-url "http://127.0.0.1:${PORT}" \
  --limit "${LIMIT:-16}" \
  --out "$OUT"

echo "Wrote rescored rows to $OUT"

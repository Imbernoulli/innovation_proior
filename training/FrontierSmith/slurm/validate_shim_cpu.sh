#!/usr/bin/env bash
# Fidelity validation of the userns-free judge backend (GJ_BACKEND=shim):
# starts the judge stack with the shim, then re-scores banked pre-outage
# FrontierCS samples and diffs per-sample against the real-go-judge scores.
#   sbatch --export=ALL,SAMPLES=<samples.jsonl>,OUT=<validation.jsonl> \
#       slurm/validate_shim_cpu.sh
#SBATCH --job-name=fs-shim-validate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
# BASH_SOURCE points at the Slurm spool copy under sbatch; hardcode the repo.
PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
cd "$PROJECT_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export TMPDIR="/tmp"

export PORT="${PORT:-$(( 18000 + (SLURM_JOB_ID % 2000) ))}"
export GJ_PORT="${GJ_PORT:-$(( 25000 + (SLURM_JOB_ID % 2000) ))}"
export GJ_BACKEND=shim
export GJ_PARALLELISM="${GJ_PARALLELISM:-16}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-12}"
export RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.cache/frontiercs-judge-shimval-${SLURM_JOB_ID:-$$}}"

SAMPLES="${SAMPLES:?need SAMPLES=<banked samples.jsonl>}"
OUT="${OUT:?need OUT=<validation output jsonl>}"
LIMIT="${LIMIT:-0}"
VAL_WORKERS="${VAL_WORKERS:-12}"

echo "PORT=$PORT GJ_PORT=$GJ_PORT backend=shim runtime=$RUNTIME_DIR"
echo "SAMPLES=$SAMPLES OUT=$OUT LIMIT=$LIMIT"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"
cleanup() {
  kill "${JUDGE_PID}" >/dev/null 2>&1 || true
  wait "${JUDGE_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

READY=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 0.5
  if ! kill -0 "${JUDGE_PID}" >/dev/null 2>&1; then
    echo "judge launcher exited early" >&2; exit 1
  fi
done
[ "$READY" = 1 ] || { echo "judge never became healthy" >&2; exit 1; }
echo "judge health:"; curl -fsS "http://127.0.0.1:${PORT}/health"; echo

FRONTIERCS_JUDGE_FAIL_SOFT=0 \
.venv/bin/python scripts/validate_gojudge_shim.py \
  --samples "$SAMPLES" \
  --judge-url "http://127.0.0.1:${PORT}" \
  --workers "$VAL_WORKERS" \
  --limit "$LIMIT" \
  --out "$OUT"

echo "DONE -> $OUT"

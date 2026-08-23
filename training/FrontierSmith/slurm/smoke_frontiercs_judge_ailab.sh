#!/usr/bin/env bash
#SBATCH --job-name=fs-judge-smoke
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PORT="${PORT:-8082}"
export GJ_PORT="${GJ_PORT:-5050}"
export GJ_PARALLELISM="${GJ_PARALLELISM:-2}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-2}"
export RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.cache/frontiercs-judge-slurm-smoke}"

scripts/start_frontiercs_judge_hybrid.sh &
JUDGE_PID="$!"

cleanup() {
  kill "${JUDGE_PID}" >/dev/null 2>&1 || true
  wait "${JUDGE_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "${JUDGE_PID}" >/dev/null 2>&1; then
    echo "judge launcher exited early" >&2
    exit 1
  fi
done

curl -fsS "http://127.0.0.1:${PORT}/health"
echo

CODE='#include <bits/stdc++.h>
using namespace std;
int main(){return 0;}
'

RESP="$(
  jq -n --arg pid frontiersmith_1 --arg lang cpp --arg code "${CODE}" \
    '{pid:$pid,lang:$lang,code:$code}' \
  | curl -fsS -X POST "http://127.0.0.1:${PORT}/submit" \
      -H 'Content-Type: application/json' \
      --data-binary @-
)"
echo "submit: ${RESP}"

SID="$(printf '%s' "${RESP}" | jq -r '.sid')"
FINAL=""
for i in $(seq 1 120); do
  FINAL="$(curl -fsS "http://127.0.0.1:${PORT}/result/${SID}")"
  STATUS="$(printf '%s' "${FINAL}" | jq -r '.status // empty')"
  echo "poll ${i}: ${STATUS}"
  case "${STATUS}" in
    queued|pending|running|'') sleep 1 ;;
    *) break ;;
  esac
done

printf '%s\n' "${FINAL}" | jq '{status, passed, result, score, cases: (.cases | length)}'
test "$(printf '%s' "${FINAL}" | jq -r '.status')" = "done"

#!/usr/bin/env bash
# V2 starter (timing fidelity, 2026-08-19): identical to
# start_frontiercs_judge_hybrid.sh except the shim backend is SELECTABLE via
#   SHIM_BIN  (default scripts/gojudge_shim_v2.py -- v1 + optional per-run core
#             pinning via SHIM_PIN_CORES / SHIM_CORES_PER_RUN; with
#             SHIM_PIN_CORES unset v2 behaves byte-identically to v1).
# New file so no running job's judge stack changes under it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PORT:=8082}"
: "${GJ_PORT:=5050}"
: "${GJ_PARALLELISM:=8}"
: "${JUDGE_WORKERS:=8}"
: "${SIF:=${REPO_ROOT}/.cache/apptainer/frontiercs-judge.sif}"
: "${GO_JUDGE_BIN:=${REPO_ROOT}/.cache/bin/go-judge}"
: "${PROBLEMS_DIR:=${REPO_ROOT}/Frontier-CS/algorithmic/problems}"
: "${RUNTIME_DIR:=${REPO_ROOT}/.cache/frontiercs-judge/${PORT}}"
: "${LOG_DIR:=${RUNTIME_DIR}/logs}"
# go-judge keeps its file store under a fixed /dev/shm/go-judge (tmpfs/RAM) path
# by default. That has two failure modes: (a) two co-located instances collide on
# the shared path, and (b) the tmpfs store consumes RAM that, under memory
# pressure during co-located vLLM warmup, gets go-judge OOM-killed (SIGKILL) right
# after it binds its port -- it serves /version but never /run, which silently
# zeroed whole evals. Use a UNIQUE, DISK-backed work dir (under TMPDIR, not
# /dev/shm) so each job is isolated and go-judge's store does not eat RAM.
: "${GJ_DIR:=${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX:-${SLURM_JOB_ID:-$$}}}"
# Sandbox backend:
#   gojudge (default) -- the real go-judge binary; needs unprivileged userns.
#   shim              -- scripts/gojudge_shim.py, API-compatible, needs NO
#                        userns; rlimit semantics matching go-judge's observed
#                        cgroup-less fallback on this cluster. Validated
#                        against pre-outage banked scores before use.
#   auto              -- gojudge when `unshare -U true` works, else shim.
# Default auto since 2026-08-01: shim CERTIFIED on ailab (853-sample regression
# vs pre-outage go-judge scores: 94.3% byte-exact, avg@5 delta -0.07, no
# systematic bias) and userns is assumed permanently disabled on della.
: "${GJ_BACKEND:=auto}"
if [[ "${GJ_BACKEND}" == "auto" ]]; then
  if unshare -U true >/dev/null 2>&1; then GJ_BACKEND=gojudge; else GJ_BACKEND=shim; fi
  echo "GJ_BACKEND=auto resolved to: ${GJ_BACKEND}" >&2
fi

DATA_DIR="${RUNTIME_DIR}/data"
SUBMISSIONS_DIR="${RUNTIME_DIR}/submissions"
GJ_LOG="${LOG_DIR}/gojudge.log"
NODE_LOG="${LOG_DIR}/node.log"

mkdir -p "${DATA_DIR}" "${SUBMISSIONS_DIR}" "${LOG_DIR}"

if [[ "${GJ_BACKEND}" == "gojudge" && ! -x "${GO_JUDGE_BIN}" ]]; then
  echo "Missing executable go-judge at ${GO_JUDGE_BIN}" >&2
  echo "Build or download it on a login node before submitting offline compute jobs." >&2
  exit 1
fi

if [[ ! -f "${SIF}" ]]; then
  echo "Missing Apptainer image at ${SIF}" >&2
  echo "Build it on a login node before submitting offline compute jobs." >&2
  exit 1
fi

GJPID=""
NODEPID=""

cleanup() {
  if [[ -n "${NODEPID}" ]]; then
    kill "${NODEPID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GJPID}" ]]; then
    kill "${GJPID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NODEPID}" ]]; then
    wait "${NODEPID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GJPID}" ]]; then
    wait "${GJPID}" >/dev/null 2>&1 || true
  fi
  rm -rf "${GJ_DIR}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# go-judge's container = clone(... |CLONE_NEWUSER|...) on /proc/self/exe. Without
# unprivileged user namespaces that clone returns ENOSPC and go-judge dies with
# "fork/exec /proc/self/exe: no space left on device" -- a disk-shaped message for
# a namespace problem. Say so plainly instead of letting callers hunt for a full
# filesystem. (della disabled it cluster-wide via /etc/sysctl.d/userns.conf on
# 2026-07-28; apptainer still works because its starter is setuid.)
if [[ "${GJ_BACKEND}" == "gojudge" && "${SKIP_USERNS_PREFLIGHT:-0}" != "1" ]] && ! unshare -U true >/dev/null 2>&1; then
  echo "go-judge cannot start on $(hostname): unprivileged user namespaces are disabled" >&2
  echo "  user.max_user_namespaces = $(cat /proc/sys/user/max_user_namespaces 2>/dev/null)" >&2
  echo "  repro: unshare -U true  ->  'unshare failed: No space left on device'" >&2
  echo "  this is NOT a disk-space problem; it needs a sysadmin sysctl change." >&2
  echo "  (or run the userns-free backend: GJ_BACKEND=shim)" >&2
  exit 1
fi

mkdir -p "${GJ_DIR}"
if [[ "${GJ_BACKEND}" == "shim" ]]; then
  echo "=== SANDBOX BACKEND: gojudge_shim.py (userns-free) -- NOT the real go-judge ===" >&2
  : "${SHIM_BIN:=${SCRIPT_DIR}/gojudge_shim_v2.py}"
  echo "SHIM_BIN=${SHIM_BIN} SHIM_PIN_CORES=${SHIM_PIN_CORES:-<unset>}" >&2
  python3 "${SHIM_BIN}" \
    -parallelism "${GJ_PARALLELISM}" \
    -http-addr "127.0.0.1:${GJ_PORT}" \
    -dir "${GJ_DIR}" \
    >"${GJ_LOG}" 2>&1 &
else
"${GO_JUDGE_BIN}" \
  -parallelism "${GJ_PARALLELISM}" \
  -http-addr "127.0.0.1:${GJ_PORT}" \
  -dir "${GJ_DIR}" \
  -cgroup-prefix "${GJ_CGROUP_PREFIX:-gojudge-${SLURM_JOB_ID:-$$}}" \
  >"${GJ_LOG}" 2>&1 &
fi
GJPID="$!"

for _ in $(seq 1 80); do
  if curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "${GJPID}" >/dev/null 2>&1; then
    echo "go-judge exited before becoming healthy; log follows:" >&2
    sed -n '1,160p' "${GJ_LOG}" >&2
    exit 1
  fi
done

if ! curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1; then
  echo "Timed out waiting for go-judge at 127.0.0.1:${GJ_PORT}; log follows:" >&2
  sed -n '1,160p' "${GJ_LOG}" >&2
  exit 1
fi

apptainer exec --cleanenv \
  --env "PORT=${PORT},GJ_ADDR=http://127.0.0.1:${GJ_PORT},JUDGE_WORKERS=${JUDGE_WORKERS},SUBMISSIONS_DIR=/app/submissions,TESTLIB_INSIDE=/app/include" \
  --bind "${REPO_ROOT}/Frontier-CS/algorithmic/server.js:/app/server.js:ro" \
  --bind "${JUDGE_SRC_DIR:-${REPO_ROOT}/Frontier-CS/algorithmic/judge/src}:/app/src:ro" \
  --bind "${REPO_ROOT}/Frontier-CS/algorithmic/judge/include:/app/include:ro" \
  --bind "${REPO_ROOT}/Frontier-CS/algorithmic/judge/config:/app/config:ro" \
  --bind "${PROBLEMS_DIR}:/app/problems:ro" \
  --bind "${DATA_DIR}:/app/data" \
  --bind "${SUBMISSIONS_DIR}:/app/submissions" \
  "${SIF}" node /app/server.js \
  >"${NODE_LOG}" 2>&1 &
NODEPID="$!"

for _ in $(seq 1 80); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Frontier-CS judge ready at http://127.0.0.1:${PORT} (go-judge http://127.0.0.1:${GJ_PORT}, dir ${GJ_DIR})"
    # Monitor BOTH processes. go-judge silently dying (binds port, serves
    # /version, then exits before the first /run) is the failure that made whole
    # evals score 0 via ECONNREFUSED. If either process exits, tear down so the
    # wrapper's health gate fails loudly instead of scoring against a dead judge.
    while kill -0 "${NODEPID}" >/dev/null 2>&1 && kill -0 "${GJPID}" >/dev/null 2>&1; do
      sleep 2
    done
    if ! kill -0 "${GJPID}" >/dev/null 2>&1; then
      echo "go-judge process exited (was on 127.0.0.1:${GJ_PORT}); log tail:" >&2
      tail -n 40 "${GJ_LOG}" >&2
    fi
    if ! kill -0 "${NODEPID}" >/dev/null 2>&1; then
      echo "Node judge API process exited (was on 127.0.0.1:${PORT}); log tail:" >&2
      tail -n 40 "${NODE_LOG}" >&2
    fi
    exit 1
  fi
  sleep 0.5
  if ! kill -0 "${NODEPID}" >/dev/null 2>&1; then
    echo "Node judge API exited before becoming healthy; log follows:" >&2
    sed -n '1,160p' "${NODE_LOG}" >&2
    exit 1
  fi
done

echo "Timed out waiting for Frontier-CS judge API at 127.0.0.1:${PORT}; log follows:" >&2
sed -n '1,160p' "${NODE_LOG}" >&2
exit 1

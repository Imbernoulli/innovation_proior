#!/usr/bin/env bash
# gpublaze replacement for scripts/start_frontiercs_judge_hybrid{,_v2}.sh.
#
# The Princeton version ran: go-judge (or the python shim) on the HOST, plus the
# Node judge API inside an Apptainer SIF (the SIF only supplied a node runtime;
# server.js/src/include/config/problems were all bind-mounted from the repo, and
# compilation always happened on the HOST via go-judge/shim).
#
# gpublaze has no Apptainer but has node v25 on the host, so the equivalent is:
#   - go-judge: the REAL binary (userns works here; .cache/bin/go-judge), or the
#     certified shim (GJ_BACKEND=shim -> scripts/gojudge_shim_v2.py). Default auto.
#   - Node judge API: run server.js directly on the host from a STAGE dir that
#     recreates the /app layout via symlinks + one `npm ci` (server.js resolves
#     problems/data relative to its own dir and has no env override for them).
# A docker-based alternative (build Frontier-CS/algorithmic/Dockerfile, run with
# --network host and the same binds) is documented in docs/EVAL_ON_GPUBLAZE_zh.md.
#
# Interface-compatible with the historical starter: PORT (node API, default 8082),
# GJ_PORT (go-judge, 5050), GJ_PARALLELISM, JUDGE_WORKERS, PROBLEMS_DIR,
# RUNTIME_DIR, GJ_BACKEND (auto|gojudge|shim), SHIM_BIN, SHIM_PIN_CORES.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_gpublaze.sh"
REPO_ROOT="$FS_ROOT"

: "${PORT:=8082}"
: "${GJ_PORT:=5050}"
: "${GJ_PARALLELISM:=8}"
: "${JUDGE_WORKERS:=8}"
: "${PROBLEMS_DIR:=${REPO_ROOT}/Frontier-CS/algorithmic/problems}"
: "${RUNTIME_DIR:=${REPO_ROOT}/.cache/frontiercs-judge/${PORT}}"
: "${LOG_DIR:=${RUNTIME_DIR}/logs}"
: "${GJ_DIR:=${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX:-$$}}"
STAGE_DIR="${STAGE_DIR:-${REPO_ROOT}/.cache/gpublaze/judge_app}"
NODE_BIN="${NODE_BIN:-$(command -v node || echo /home/bohanlyu/.nvm/versions/node/v25.9.0/bin/node)}"
NPM_BIN="${NPM_BIN:-$(command -v npm || echo /home/bohanlyu/.nvm/versions/node/v25.9.0/bin/npm)}"

if [[ "${GJ_BACKEND:=auto}" == "auto" ]]; then
  if unshare -U true >/dev/null 2>&1 && [ -x "$GO_JUDGE_BIN" ]; then GJ_BACKEND=gojudge; else GJ_BACKEND=shim; fi
  echo "GJ_BACKEND=auto resolved to: ${GJ_BACKEND}" >&2
fi

[ -d "$PROBLEMS_DIR" ] || { echo "Missing PROBLEMS_DIR=$PROBLEMS_DIR (run scripts/gpublaze/prepare_assets_gpublaze.sh)" >&2; exit 1; }

# ---- stage the node app dir (one-time; symlinks track the repo) ---------------
# Serialized under flock: concurrent judge starts (e.g. two arms' eval clients
# launching at once) raced on `rm -rf src; cp -r ... src` and died with
# "cp: cannot create directory .../src: File exists".
mkdir -p "$STAGE_DIR"
exec 9>"$STAGE_DIR/.stage.lock"
flock 9
cp -f "$REPO_ROOT/Frontier-CS/algorithmic/server.js" "$STAGE_DIR/server.js"
cp -f "$REPO_ROOT/Frontier-CS/algorithmic/package.json" "$STAGE_DIR/package.json"
cp -f "$REPO_ROOT/Frontier-CS/algorithmic/package-lock.json" "$STAGE_DIR/package-lock.json"
# src must be a COPY, not a symlink: node's ESM loader realpaths modules, so a
# symlinked src would resolve its own imports (express/js-yaml/...) relative to
# the repo tree, where there is no node_modules. The tree is 6 small files;
# refresh it on every start so repo edits are picked up.
rm -rf "$STAGE_DIR/src"
cp -r "${JUDGE_SRC_DIR:-$REPO_ROOT/Frontier-CS/algorithmic/judge/src}" "$STAGE_DIR/src"
ln -sfn "$REPO_ROOT/Frontier-CS/algorithmic/judge/include" "$STAGE_DIR/include"
ln -sfn "$REPO_ROOT/Frontier-CS/algorithmic/judge/config" "$STAGE_DIR/config"
ln -sfn "$PROBLEMS_DIR" "$STAGE_DIR/problems"
DATA_DIR="${RUNTIME_DIR}/data"; SUBMISSIONS_DIR="${RUNTIME_DIR}/submissions"
mkdir -p "$DATA_DIR" "$SUBMISSIONS_DIR" "$LOG_DIR" "$GJ_DIR"
ln -sfn "$DATA_DIR" "$STAGE_DIR/data"
if [ ! -d "$STAGE_DIR/node_modules" ]; then
  echo "[judge-local] npm ci (first run) ..." >&2
  (cd "$STAGE_DIR" && "$NPM_BIN" ci --omit=dev --ignore-scripts >/dev/null)
fi
flock -u 9

GJ_LOG="${LOG_DIR}/gojudge.log"; NODE_LOG="${LOG_DIR}/node.log"
GJPID=""; NODEPID=""
cleanup() {
  [ -n "$NODEPID" ] && kill "$NODEPID" >/dev/null 2>&1 || true
  [ -n "$GJPID" ]   && kill "$GJPID"   >/dev/null 2>&1 || true
  [ -n "$NODEPID" ] && wait "$NODEPID" >/dev/null 2>&1 || true
  [ -n "$GJPID" ]   && wait "$GJPID"   >/dev/null 2>&1 || true
  rm -rf "$GJ_DIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ "$GJ_BACKEND" == "shim" ]]; then
  echo "=== SANDBOX BACKEND: gojudge_shim (userns-free) -- NOT the real go-judge ===" >&2
  : "${SHIM_BIN:=${REPO_ROOT}/scripts/gojudge_shim_v2.py}"
  echo "SHIM_BIN=${SHIM_BIN} SHIM_PIN_CORES=${SHIM_PIN_CORES:-<unset>}" >&2
  "$FS_CLIENT_VENV/bin/python" "$SHIM_BIN" \
    -parallelism "$GJ_PARALLELISM" -http-addr "127.0.0.1:${GJ_PORT}" -dir "$GJ_DIR" \
    >"$GJ_LOG" 2>&1 &
else
  "$GO_JUDGE_BIN" \
    -parallelism "$GJ_PARALLELISM" -http-addr "127.0.0.1:${GJ_PORT}" -dir "$GJ_DIR" \
    -cgroup-prefix "${GJ_CGROUP_PREFIX:-gojudge-$$}" \
    >"$GJ_LOG" 2>&1 &
fi
GJPID="$!"

for _ in $(seq 1 80); do
  curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 && break
  sleep 0.5
  kill -0 "$GJPID" >/dev/null 2>&1 || { echo "go-judge/shim exited before healthy; log:" >&2; sed -n '1,80p' "$GJ_LOG" >&2; exit 1; }
done
curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 || { echo "timeout waiting go-judge on $GJ_PORT" >&2; exit 1; }

PORT="$PORT" GJ_ADDR="http://127.0.0.1:${GJ_PORT}" JUDGE_WORKERS="$JUDGE_WORKERS" \
SUBMISSIONS_DIR="$SUBMISSIONS_DIR" TESTLIB_INSIDE="$STAGE_DIR/include" \
  "$NODE_BIN" "$STAGE_DIR/server.js" >"$NODE_LOG" 2>&1 &
NODEPID="$!"

for _ in $(seq 1 80); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Frontier-CS judge ready at http://127.0.0.1:${PORT} (go-judge http://127.0.0.1:${GJ_PORT}, backend=$GJ_BACKEND, dir $GJ_DIR)"
    while kill -0 "$NODEPID" >/dev/null 2>&1 && kill -0 "$GJPID" >/dev/null 2>&1; do sleep 2; done
    kill -0 "$GJPID"  >/dev/null 2>&1 || { echo "go-judge exited; log tail:" >&2; tail -n 40 "$GJ_LOG" >&2; }
    kill -0 "$NODEPID" >/dev/null 2>&1 || { echo "node judge exited; log tail:" >&2; tail -n 40 "$NODE_LOG" >&2; }
    exit 1
  fi
  sleep 0.5
  kill -0 "$NODEPID" >/dev/null 2>&1 || { echo "node judge exited before healthy; log:" >&2; sed -n '1,80p' "$NODE_LOG" >&2; exit 1; }
done
echo "timeout waiting node judge on $PORT; log:" >&2; sed -n '1,80p' "$NODE_LOG" >&2; exit 1

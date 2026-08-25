#!/usr/bin/env bash
# jiaolab port of scripts/gpublaze/start_frontiercs_judge_local.sh.
#
# Identical topology (real go-judge on the host + the Node judge API run
# directly from a staged /app layout). Machine deltas:
#   - STAGE_DIR defaults to .cache/jiaolab/judge_app (node_modules rsynced from
#     gpublaze; pure-JS deps, so they are node-version portable).
#   - node here is v18 (gpublaze has v25); server.js is plain ESM + express +
#     js-yaml, all of which run on 18.
#   - userns works (`unshare -U true` passes, unprivileged_userns_clone=1), so
#     GJ_BACKEND=auto resolves to the REAL go-judge binary, exactly like gpublaze.
#     GJ_BACKEND=shim falls back to the certified gojudge_shim_v2.py.
#
# Interface-compatible with the historical starter: PORT (node API, default 8082),
# GJ_PORT (go-judge, 5050), GJ_PARALLELISM, JUDGE_WORKERS, PROBLEMS_DIR,
# RUNTIME_DIR, GJ_BACKEND (auto|gojudge|shim), SHIM_BIN, SHIM_PIN_CORES.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"
REPO_ROOT="$FS_ROOT"

: "${PORT:=8082}"
: "${GJ_PORT:=5050}"
: "${GJ_PARALLELISM:=8}"
: "${JUDGE_WORKERS:=8}"
: "${PROBLEMS_DIR:=${REPO_ROOT}/Frontier-CS/algorithmic/problems}"
: "${RUNTIME_DIR:=${REPO_ROOT}/.cache/frontiercs-judge/${PORT}}"
: "${LOG_DIR:=${RUNTIME_DIR}/logs}"
: "${GJ_DIR:=${TMPDIR:-/tmp}/go-judge-${GJ_CGROUP_PREFIX:-$$}}"
STAGE_DIR="${STAGE_DIR:-$FS_JUDGE_STAGE_DIR}"
NODE_BIN="${NODE_BIN:-$(command -v node)}"
NPM_BIN="${NPM_BIN:-$(command -v npm)}"

# NOTE (jiaolab, verified 2026-08-25): the probe is `unshare -Ur`, NOT gpublaze's
# `unshare -U`. This box is Ubuntu 24.04 / kernel 6.8 with AppArmor's
# `apparmor_restrict_unprivileged_userns`: creating an empty user namespace
# succeeds (`unshare -U true` passes) but writing /proc/self/uid_map is denied, so
# the real go-judge dies at startup with
#   container: failed to start container: fork/exec /proc/self/exe: permission denied
# Lifting that needs sudo (AppArmor profile / sysctl), which we do not have --
# running go-judge inside apptainer does NOT help (nested userns is denied too).
# Hence jiaolab judges with the certified gojudge_shim_v2 (Princeton-authenticated
# 94.3% byte-exact vs the real go-judge). This is recorded in every run's
# judge_node_meta.json as judge_variant=jiaolab_local_shim.
if [[ "${GJ_BACKEND:=auto}" == "auto" ]]; then
  if unshare -Ur true >/dev/null 2>&1 && [ -x "$GO_JUDGE_BIN" ]; then GJ_BACKEND=gojudge; else GJ_BACKEND=shim; fi
  echo "GJ_BACKEND=auto resolved to: ${GJ_BACKEND}" >&2
fi

[ -d "$PROBLEMS_DIR" ] || { echo "Missing PROBLEMS_DIR=$PROBLEMS_DIR (rsync the official Frontier-CS problems from gpublaze)" >&2; exit 1; }

# ---- stage the node app dir (one-time; symlinks track the repo) ---------------
mkdir -p "$STAGE_DIR"
exec 9>"$STAGE_DIR/.stage.lock"
flock 9
cp -f "$REPO_ROOT/Frontier-CS/algorithmic/server.js" "$STAGE_DIR/server.js"
cp -f "$REPO_ROOT/Frontier-CS/algorithmic/package.json" "$STAGE_DIR/package.json"
cp -f "$REPO_ROOT/Frontier-CS/algorithmic/package-lock.json" "$STAGE_DIR/package-lock.json"
# src must be a COPY, not a symlink: node's ESM loader realpaths modules.
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

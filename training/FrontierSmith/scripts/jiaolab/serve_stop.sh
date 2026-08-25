#!/usr/bin/env bash
# Stop a serve_local.sh backend cleanly, by TAG. Only kills PIDs recorded in the
# registry pidfiles WE wrote, and only if the process still belongs to $USER --
# never anything else on the machine.
#
#   bash scripts/gpublaze/serve_stop.sh <TAG>       # stop backends with this tag
#   bash scripts/gpublaze/serve_stop.sh --list      # show registry state
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_jiaolab.sh"

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
  echo "registry: $VLLM_POOL_REGISTRY"
  for f in "$VLLM_POOL_REGISTRY"/*.json; do
    [ -f "$f" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    echo "  $(basename "$f")  (heartbeat ${age}s ago)"
  done
  exit 0
fi

TAG="$1"; found=0
for pf in "$VLLM_POOL_REGISTRY/${TAG}__"*.pid; do
  [ -f "$pf" ] || continue
  pid=$(cat "$pf")
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$owner" ] && [ "$owner" != "$USER" ]; then
    echo "REFUSING to kill pid $pid: owned by $owner, not $USER" >&2; continue
  fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "[serve-stop] TERM process group $pid ($TAG)"
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -0 "$pid" 2>/dev/null && { echo "[serve-stop] KILL $pid"; kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null; }
  fi
  rm -f "$pf" "${pf%.pid}.json"; rm -rf "${pf%.pid}.json.clients"
  found=1
done
[ "$found" = 1 ] || echo "[serve-stop] no pidfile for TAG=$TAG in $VLLM_POOL_REGISTRY"

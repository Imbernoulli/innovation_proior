#!/usr/bin/env bash
# jiaolab port of scripts/apply_vllm_penalty_fastpath.sh: same injected block,
# but targeting THIS machine's RL venv (/home/bohan/venv-vllm-jiaolab, vLLM 0.21.0) and this
# repo's scripts dir. Verified 2026-08-23: vllm 0.21's
# vllm/v1/sample/ops/penalties.py has the identical _convert_to_tensors
# signature the fast path replaces. The block stays INERT unless
# FS_VLLM_PENALTY_FASTPATH=1.
#   apply:  scripts/jiaolab/apply_vllm_penalty_fastpath_jiaolab.sh
#   revert: scripts/jiaolab/apply_vllm_penalty_fastpath_jiaolab.sh --revert
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"
VENV="${RL_VENV:-${VLLM_VENV:-/home/bohan/venv-vllm-jiaolab}}"
F="$VENV/lib/python3.12/site-packages/vllm/v1/sample/ops/penalties.py"
MARK="# --- FrontierSmith penalty fast path"

[ -f "$F" ] || { echo "ERROR: $F not found (venv built?)" >&2; exit 1; }

if [ "${1:-}" = "--revert" ]; then
  [ -f "$F.orig" ] && cp "$F.orig" "$F" && echo "reverted from $F.orig" || echo "no backup found"
  exit 0
fi
grep -q "$MARK" "$F" && { echo "already installed"; exit 0; }
[ -f "$F.orig" ] || cp "$F" "$F.orig"
cat >> "$F" <<PYEOF


# --- FrontierSmith penalty fast path (gpublaze port; see scripts/vllm_penalty_fastpath.py)
import os as _fs_os

if _fs_os.environ.get("FS_VLLM_PENALTY_FASTPATH", "0").strip().lower() in ("1", "true", "yes"):
    try:
        import sys as _fs_sys

        _fs_dir = "$FS_ROOT/scripts"
        if _fs_dir not in _fs_sys.path:
            _fs_sys.path.insert(0, _fs_dir)
        import vllm_penalty_fastpath as _fs_fp

        _fs_original_convert = _convert_to_tensors
        _convert_to_tensors = _fs_fp._fast_convert
        _fs_fastpath_installed = True
        print("[fs] vLLM penalty fast path ACTIVE", flush=True)
    except Exception as _fs_e:  # noqa: BLE001
        print(f"[fs] vLLM penalty fast path FAILED, using stock: {_fs_e!r}", flush=True)
PYEOF
echo "installed into $F (backup: $F.orig)"

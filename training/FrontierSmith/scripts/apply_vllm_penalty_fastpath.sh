#!/usr/bin/env bash
# Install (or revert) the penalty fast path into the venv's vLLM.
#   apply:  scripts/apply_vllm_penalty_fastpath.sh
#   revert: scripts/apply_vllm_penalty_fastpath.sh --revert
# The injected block is INERT unless FS_VLLM_PENALTY_FASTPATH=1, so the venv
# behaves exactly as stock for anything that does not opt in.
set -euo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
F="$FS/.venv-vllm023/lib/python3.12/site-packages/vllm/v1/sample/ops/penalties.py"
MARK="# --- FrontierSmith penalty fast path"

if [ "${1:-}" = "--revert" ]; then
  [ -f "$F.orig" ] && cp "$F.orig" "$F" && echo "reverted from $F.orig" || echo "no backup found"
  exit 0
fi
grep -q "$MARK" "$F" && { echo "already installed"; exit 0; }
[ -f "$F.orig" ] || cp "$F" "$F.orig"
cat >> "$F" <<'PYEOF'


# --- FrontierSmith penalty fast path (2026-08-13) -----------------------------
# _convert_to_tensors above rebuilds a [batch, max_len] int64 tensor every decode
# step; measured 162 ms/step at B=128 L=32768, ~70% of our decode time (py-spy
# caught the sampler here in 12/14 samples, GPUs idling at 0-37%). The
# replacement appends only the new tokens and is bit-identical (selftest in
# scripts/vllm_penalty_fastpath.py). Inert unless FS_VLLM_PENALTY_FASTPATH=1.
import os as _fs_os

if _fs_os.environ.get("FS_VLLM_PENALTY_FASTPATH", "0").strip().lower() in ("1", "true", "yes"):
    try:
        import sys as _fs_sys

        _fs_dir = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/scripts"
        if _fs_dir not in _fs_sys.path:
            _fs_sys.path.insert(0, _fs_dir)
        import vllm_penalty_fastpath as _fs_fp

        _fs_original_convert = _convert_to_tensors   # keep the stock impl so
        # selftest() can compare against it (never fast-vs-fast)
        _convert_to_tensors = _fs_fp._fast_convert
        _fs_fastpath_installed = True
        print("[fs] vLLM penalty fast path ACTIVE", flush=True)
    except Exception as _fs_e:  # noqa: BLE001
        print(f"[fs] vLLM penalty fast path FAILED, using stock: {_fs_e!r}", flush=True)
PYEOF
echo "installed into $F (backup: $F.orig)"

#!/usr/bin/env python3
"""Portability shim for scripts/build_alebench_full40.py on gpublaze.

The historical script hardcodes FS = /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
as a module-level constant with no env override. Per the migration rule
(override via wrappers, never edit history), this shim imports the original
module, rewrites its path constants to the local FrontierSmith root (or
$FS_ROOT), and runs its unchanged main().
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

FS_ROOT = Path(os.environ.get("FS_ROOT", Path(__file__).resolve().parent.parent.parent))
ORIG = FS_ROOT / "scripts" / "build_alebench_full40.py"

spec = importlib.util.spec_from_file_location("build_alebench_full40", ORIG)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.FS = FS_ROOT
mod.SNAP = FS_ROOT / ".cache/ale-bench/datasets--SakanaAI--ALE-Bench/snapshots"
mod.REFS = FS_ROOT / "data/alebench/ale_score_references.json"
mod.VAL = FS_ROOT / "data/alebench/val.parquet"

if __name__ == "__main__":
    if not any(a.startswith("--out") for a in sys.argv[1:]):
        sys.argv += ["--out", str(FS_ROOT / "data/alebench/full40.parquet")]
    raise SystemExit(mod.main())

#!/usr/bin/env python3
"""Prebuild ALE-Bench Rust judge tools for offline compute nodes.

Run this on a networked login node before using ALE rewards in Slurm jobs.
Compute nodes can then restore cached binaries without contacting crates.io.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TOOLS = ("gen", "tester", "vis")


def required_tools(tool_dir: Path) -> list[str]:
    return [tool for tool in TOOLS if (tool_dir / "src" / "bin" / f"{tool}.rs").is_file()]


def cached_tools(cache_root: Path, cache_key: str, tools: list[str]) -> list[str]:
    cache_dir = cache_root / cache_key
    return [tool for tool in tools if (cache_dir / tool).is_file()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "alebench" / "local_data")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "ale-bench")
    parser.add_argument("--apptainer-dir", type=Path, default=PROJECT_ROOT / ".cache" / "apptainer" / "alebench")
    parser.add_argument("--lite", dest="lite", action="store_true", default=True)
    parser.add_argument("--no-lite", dest="lite", action="store_false")
    parser.add_argument("--problem-id", action="append", default=None, help="Only prepare selected problem id(s).")
    parser.add_argument("--check-only", action="store_true", help="Report missing cached tools without building.")
    args = parser.parse_args()

    os.environ["ALE_BENCH_DATA"] = str(args.data_dir.resolve())
    os.environ["ALE_BENCH_CACHE"] = str(args.cache_dir.resolve())
    os.environ["ALE_BENCH_TOOL_CACHE"] = str((args.cache_dir / "rust-tool-builds").resolve())
    os.environ["ALE_BENCH_CONTAINER_BACKEND"] = "apptainer"
    os.environ["ALE_BENCH_APPTAINER_DIR"] = str(args.apptainer_dir.resolve())

    sys.path.insert(0, str(PROJECT_ROOT / "ALE-Bench" / "src"))
    from ale_bench.data import _rust_tool_cache_key, build_rust_tools, list_problem_ids, load_problem

    problem_ids = args.problem_id or list_problem_ids(lite_version=args.lite)
    known = set(list_problem_ids(lite_version=args.lite))
    unknown = sorted(set(problem_ids) - known)
    if unknown:
        raise SystemExit(f"Unknown problem id(s) for lite={args.lite}: {', '.join(unknown)}")

    cache_root = Path(os.environ["ALE_BENCH_TOOL_CACHE"])
    prepared = 0
    missing: list[str] = []
    for problem_id in problem_ids:
        _problem, _seeds, _standings, _rpm, data_root = load_problem(problem_id, lite_version=args.lite)
        tool_dir = data_root / "tools"
        tools = required_tools(tool_dir)
        cache_key = _rust_tool_cache_key(tool_dir)
        present = cached_tools(cache_root, cache_key, tools)
        if len(present) == len(tools):
            print(f"OK cached {problem_id} {cache_key} tools={','.join(tools) or '-'}")
            prepared += 1
            continue

        missing_tools = sorted(set(tools) - set(present))
        if args.check_only:
            print(f"MISSING {problem_id} {cache_key} tools={','.join(missing_tools)}")
            missing.append(problem_id)
            continue

        build_rust_tools(tool_dir)
        present = cached_tools(cache_root, cache_key, tools)
        if len(present) != len(tools):
            raise RuntimeError(f"Cache incomplete for {problem_id}: expected {tools}, got {present}")
        print(f"BUILT {problem_id} {cache_key} tools={','.join(tools) or '-'}")
        prepared += 1

    if missing:
        raise SystemExit(f"\nMissing cached Rust tools for {len(missing)} problems: {', '.join(missing)}")
    print(f"\nPrepared Rust tool cache for {prepared}/{len(problem_ids)} problems: {os.environ['ALE_BENCH_TOOL_CACHE']}")


if __name__ == "__main__":
    main()

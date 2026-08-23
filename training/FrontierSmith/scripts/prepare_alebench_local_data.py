#!/usr/bin/env python3
"""Materialize ALE-Bench data files for offline compute-node evaluation.

The ALE-Bench package can read local data when ALE_BENCH_DATA points at a
directory containing problem id lists and per-problem zip files. This script
downloads those files on a networked login node and copies them into the repo.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "SakanaAI/ALE-Bench"
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "alebench" / "local_data"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "ale-bench"


def download_file(filename: str, output_dir: Path, cache_dir: Path) -> Path:
    src = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )
    dst = output_dir / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def read_problem_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--lite", dest="lite", action="store_true", default=True)
    parser.add_argument("--no-lite", dest="lite", action="store_false")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    sidecars = ["problem_ids_lite.txt", "problem_ids.txt", "schedule.csv", "ranking.csv"]
    for filename in sidecars:
        try:
            path = download_file(filename, args.output_dir, args.cache_dir)
            print(f"OK {filename} -> {path}")
        except Exception as exc:
            print(f"SKIP {filename}: {exc}")

    id_file = args.output_dir / ("problem_ids_lite.txt" if args.lite else "problem_ids.txt")
    problem_ids = read_problem_ids(id_file)
    for problem_id in problem_ids:
        path = download_file(f"{problem_id}.zip", args.output_dir, args.cache_dir)
        print(f"OK {problem_id}.zip -> {path}")

    print(f"\nSaved {len(problem_ids)} problem zips in {args.output_dir}")


if __name__ == "__main__":
    main()

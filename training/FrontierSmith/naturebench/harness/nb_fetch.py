"""nb_fetch.py — download NatureBench task packages (login node only).

Mirrors run_naturebench.py's download semantics: per-task allow_patterns, then
materialize any problem/data_archives/*.tar.gz into problem/data (safe 'data'
tar filter), removing the archive dir afterwards.

Usage:
    python nb_fetch.py --tasks-file subset.txt
    python nb_fetch.py --task s42256-022-00468-6 [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

NB_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = NB_ROOT / "data" / "naturebench_data"


def extract_archives(tasks_dir: Path, task_id: str) -> bool:
    problem_dir = tasks_dir / task_id / "problem"
    archive_dir = problem_dir / "data_archives"
    if not archive_dir.is_dir():
        return False
    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        return False
    data_dir = problem_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    for ap in archives:
        print(f"  extracting {ap.name}", flush=True)
        with tarfile.open(ap, "r:gz") as tf:
            tf.extractall(problem_dir, filter="data")
    if not data_dir.is_dir():
        raise RuntimeError(f"archives for {task_id} did not create {data_dir}")
    shutil.rmtree(archive_dir)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--dataset-id", default="FrontisAI/NatureBench")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if metadata.json exists (repairs partial downloads)")
    args = ap.parse_args()

    tasks = list(args.task)
    if args.tasks_file:
        tasks += [l.strip() for l in open(args.tasks_file) if l.strip() and not l.startswith("#")]
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        print("no tasks given", file=sys.stderr)
        return 2

    todo = tasks if args.force else [
        t for t in tasks if not (DATA_ROOT / "tasks" / t / "metadata.json").exists()]
    print(f"{len(tasks)} requested, {len(todo)} to download")
    if args.dry_run or not todo:
        for t in todo:
            print("  would fetch", t)
        return 0

    import time
    from huggingface_hub import snapshot_download
    # The Hub rate-limits (HTTP 429) aggressive multi-file fetches from a shared
    # cluster IP: download one task at a time, few workers, backoff on failure.
    failed = []
    for i, t in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {t}", flush=True)
        for attempt in range(1, 5):
            try:
                snapshot_download(
                    repo_id=args.dataset_id, repo_type="dataset",
                    local_dir=str(DATA_ROOT),
                    allow_patterns=[f"tasks/{t}/**"],
                    max_workers=2,
                )
                break
            except Exception as e:
                wait = 60 * attempt
                print(f"  attempt {attempt} failed ({type(e).__name__}); "
                      f"sleeping {wait}s", flush=True)
                if attempt == 4:
                    failed.append(t)
                else:
                    time.sleep(wait)
        if extract_archives(DATA_ROOT / "tasks", t):
            print(f"  {t}: archives materialized")
        time.sleep(3)
    if failed:
        print("FAILED TASKS:", " ".join(failed))
    # the HF metadata cache under .cache costs inodes and is not needed once
    # files are on disk; drop it to protect the fileset inode quota.
    cache = DATA_ROOT / ".cache"
    if cache.exists():
        shutil.rmtree(cache)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

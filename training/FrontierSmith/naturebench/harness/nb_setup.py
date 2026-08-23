"""nb_setup.py — LOGIN-NODE phase: build each task's container environment.

Compute nodes on Della have no outbound DNS, so the task's Dockerfile RUN
pip-installs (the official --skip-build setup phase) must run here. Each task
gets a writable Apptainer overlay; a marker records the exact setup script so
compute jobs skip the replay.

Records per task: setup seconds, overlay disk + inode cost, data size, and any
setup-extra repair applied.

Usage:
    python nb_setup.py --tasks-file task-sets/subset18.txt
    python nb_setup.py --task s42256-022-00468-6 --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from nb_run import (NB_ROOT, DEFAULT_DATA, DEFAULT_SIF, ApptainerTask,
                    build_setup_script, log)


def dir_stats(p: Path) -> tuple[int, int]:
    """(bytes, inodes) for a directory tree; cheap enough for overlays."""
    if not p.exists():
        return (0, 0)
    out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True)
    nbytes = int(out.stdout.split()[0]) if out.stdout.strip() else 0
    cnt = subprocess.run(f"find {p} | wc -l", shell=True, capture_output=True, text=True)
    return (nbytes, int(cnt.stdout.strip() or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--sif", default=str(DEFAULT_SIF))
    ap.add_argument("--results-dir", default=str(NB_ROOT / "results"))
    ap.add_argument("--setup-timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-pip-cache", action="store_true")
    ap.add_argument("--setup-extra-file", default=str(NB_ROOT / "task-sets" / "setup_extra.json"))
    ap.add_argument("--report", default=str(NB_ROOT / "task-sets" / "setup_report.json"))
    args = ap.parse_args()

    tasks = list(args.task)
    if args.tasks_file:
        tasks += [l.strip() for l in open(args.tasks_file)
                  if l.strip() and not l.startswith("#")]
    tasks = list(dict.fromkeys(tasks))

    setup_extra = {}
    if Path(args.setup_extra_file).exists():
        setup_extra = json.loads(Path(args.setup_extra_file).read_text())

    results_dir = Path(args.results_dir)
    overlay_root = results_dir / "_overlays"
    pip_cache = None if args.no_pip_cache else results_dir / "_pipcache"
    log_dir = NB_ROOT / "logs" / "setup"
    log_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    if Path(args.report).exists():
        report = json.loads(Path(args.report).read_text())

    for i, t in enumerate(tasks, 1):
        task_root = Path(args.data_dir) / t
        if not (task_root / "metadata.json").exists():
            log(f"[{i}/{len(tasks)}] {t}: NOT DOWNLOADED, skipping")
            report[t] = {"status": "not_downloaded"}
            continue
        data_bytes, data_inodes = dir_stats(task_root)
        ws = results_dir / "_setup_ws" / t
        (ws / ".home").mkdir(parents=True, exist_ok=True)
        ct = ApptainerTask(Path(args.sif), t, task_root, ws, overlay_root,
                           gpu=False, pip_cache=pip_cache)
        script, n_cmds = build_setup_script(task_root, setup_extra.get(t),
                                            use_pip_cache=pip_cache is not None)
        log(f"[{i}/{len(tasks)}] {t}: {n_cmds} RUN cmds, data={data_bytes/1e6:.1f} MB")
        t0 = time.time()
        try:
            state = ct.setup(script, args.setup_timeout, log_dir / f"{t}.log",
                             force=args.force)
            secs = round(time.time() - t0, 1)
            ov_bytes, ov_inodes = dir_stats(ct.overlay)
            report[t] = {"status": "ok", "setup_state": state, "setup_seconds": secs,
                         "n_run_cmds": n_cmds,
                         "data_mb": round(data_bytes / 1e6, 1), "data_inodes": data_inodes,
                         "overlay_mb": round(ov_bytes / 1e6, 1), "overlay_inodes": ov_inodes,
                         "setup_extra": setup_extra.get(t)}
            log(f"    OK ({state}, {secs}s, overlay {ov_bytes/1e6:.0f} MB / "
                f"{ov_inodes} inodes)")
        except Exception as e:
            secs = round(time.time() - t0, 1)
            tail = ""
            lp = log_dir / f"{t}.log"
            if lp.exists():
                lines = [l for l in lp.read_text(errors="replace").splitlines() if l.strip()]
                tail = " | ".join(lines[-3:])[:400]
            report[t] = {"status": "setup_failed", "setup_seconds": secs,
                         "n_run_cmds": n_cmds, "data_mb": round(data_bytes / 1e6, 1),
                         "error": f"{type(e).__name__}: {e}", "log_tail": tail}
            log(f"    FAILED after {secs}s: {tail[:200]}")
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))

    ok = [t for t, r in report.items() if r.get("status") == "ok"]
    bad = [t for t, r in report.items() if r.get("status") not in ("ok",)]
    log(f"setup done: {len(ok)} ok, {len(bad)} not ok")
    if ok:
        (NB_ROOT / "task-sets" / "working.txt").write_text("\n".join(sorted(ok)) + "\n")
        log(f"wrote task-sets/working.txt ({len(ok)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

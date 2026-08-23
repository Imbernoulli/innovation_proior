#!/usr/bin/env python3
"""Merge tv2..tv5 self-check results + RQ-overlap matrix into the consolidated
training_tasks/mlsbench_tv1/selfcheck_report_tv2plus.json (sibling of the tv1
report; never clobbers it).

Inputs:
  --check   one or more checker output JSONs (scripts/mlsbench_check_train_tasks.py)
  --matrix  overlap matrix JSON (scripts/mlsbench_rq_overlap_matrix.py --all --out)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = PROJECT_ROOT / "training_tasks" / "mlsbench_tv1" / "selfcheck_report_tv2plus.json"
BROKEN_ENVS = {
    "ml-anomaly-detection",
    "ml-missing-data-imputation",
    "ml-selective-deferral",
    "ml-subgroup-calibration-shift",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    by_task: dict[str, dict] = {}
    for p in args.check:
        for rec in json.loads(Path(p).read_text()):
            by_task[rec["task"]] = rec  # later files win (rerun overrides)

    matrix = {m["base"]: m for m in json.loads(Path(args.matrix).read_text())}

    out = []
    for task in sorted(by_task):
        rec = dict(by_task[task])
        base = re.sub(r"_(?:tv\d+|train)$", "", task)
        sfx = task[len(base) + 1:]
        rec["broken_env"] = base in BROKEN_ENVS
        m = matrix.get(base)
        if m:
            rec["overlap_vs_eval"] = m["vs_eval"].get(sfx)
            rec["overlap_siblings"] = {
                pair: o for pair, o in m["pairwise"].items() if sfx in pair.split("-")
            }
            rec["overlap_ok"] = m["ok"]
        out.append(rec)

    Path(args.out).write_text(json.dumps(out, indent=2))
    n_ok = sum(1 for r in out if r.get("loader_ok") and r.get("score_ok"))
    print(f"[merge] {len(out)} tasks, {n_ok} loader+score OK -> {args.out}")
    return 0


if __name__ == "__main__":
    main()

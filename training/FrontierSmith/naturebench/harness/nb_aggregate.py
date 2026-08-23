"""nb_aggregate.py — aggregate a NatureBench batch into one comparable score.

Official metric definitions (submit-results/SUBMISSION_SPEC.md):
    Match-SOTA    = share of tasks with g >= 0
    Surpass-SOTA  = share of tasks with g > 0.1
The official leaderboard divides by 90 (the full benchmark). We run a SUBSET,
so the denominator here is the subset size and every number is labelled
`subset_` to prevent it being mistaken for a full-benchmark figure.

Refuses to aggregate an incomplete subset unless --allow-partial (mirrors the
campaign's "never aggregate incomplete data" rule).

Usage:
    python nb_aggregate.py --batch mybatch --tasks-file task-sets/subset.txt
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

NB_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--results-dir", default=str(NB_ROOT / "results"))
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batch_dir = Path(args.results_dir) / args.batch
    if not batch_dir.is_dir():
        print(f"no such batch dir: {batch_dir}", file=sys.stderr)
        return 2

    if args.tasks_file:
        expected = [l.strip() for l in open(args.tasks_file)
                    if l.strip() and not l.startswith("#")]
    else:
        expected = sorted(p.name for p in batch_dir.iterdir()
                          if p.is_dir() and not p.name.startswith("_"))

    rows, missing, errored = [], [], []
    for t in expected:
        rj = batch_dir / t / "result.json"
        if not rj.exists():
            missing.append(t)
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception as e:
            errored.append((t, f"unreadable result.json: {e}"))
            continue
        if r.get("status") == "error":
            errored.append((t, r.get("error", "unknown")))
        rows.append({"task": t, "g": r.get("g"), "status": r.get("status"),
                     "attempts": r.get("total_attempts"),
                     "seconds": r.get("task_seconds"),
                     "model": r.get("model"), "mode": r.get("mode")})

    if missing and not args.allow_partial:
        print(f"REFUSING to aggregate: {len(missing)}/{len(expected)} tasks have no "
              f"result.json ({', '.join(missing[:8])}"
              f"{'...' if len(missing) > 8 else ''}).\n"
              f"Finish the batch or pass --allow-partial (results will be labelled partial).",
              file=sys.stderr)
        return 3

    scored = [r for r in rows if isinstance(r["g"], (int, float))]
    invalid = [r for r in rows if not isinstance(r["g"], (int, float))]
    n_sub = len(expected)
    gs = [r["g"] for r in scored]

    summary = {
        "batch": args.batch,
        "partial": bool(missing),
        "subset_size": n_sub,
        "tasks_with_result": len(rows),
        "tasks_scored": len(scored),
        "tasks_unscored": len(invalid),
        "tasks_missing": missing,
        "tasks_errored": [t for t, _ in errored],
        # official definitions, subset denominator
        "subset_match_sota_pct": round(100.0 * sum(g >= 0 for g in gs) / n_sub, 2) if n_sub else None,
        "subset_surpass_sota_pct": round(100.0 * sum(g > 0.1 for g in gs) / n_sub, 2) if n_sub else None,
        "mean_g_scored": round(statistics.mean(gs), 6) if gs else None,
        "median_g_scored": round(statistics.median(gs), 6) if gs else None,
        "min_g": round(min(gs), 6) if gs else None,
        "max_g": round(max(gs), 6) if gs else None,
        "total_gpu_seconds": round(sum(r["seconds"] or 0 for r in rows), 1),
        "per_task": sorted(rows, key=lambda r: (r["g"] is None, -(r["g"] or 0))),
    }

    out = Path(args.out) if args.out else batch_dir / "SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== NatureBench subset summary: {args.batch} ===")
    print(f"subset size {n_sub} | scored {len(scored)} | unscored {len(invalid)} | "
          f"missing {len(missing)} | errored {len(errored)}")
    print(f"Match-SOTA (g>=0):   {summary['subset_match_sota_pct']}%  (subset denominator)")
    print(f"Surpass-SOTA (g>0.1):{summary['subset_surpass_sota_pct']}%  (subset denominator)")
    print(f"mean g {summary['mean_g_scored']}  median {summary['median_g_scored']}  "
          f"range [{summary['min_g']}, {summary['max_g']}]")
    print(f"{'task':<24} {'g':>12}  {'att':>3} {'sec':>7}  status")
    for r in summary["per_task"]:
        g = f"{r['g']:.6f}" if isinstance(r["g"], (int, float)) else "none"
        print(f"{r['task']:<24} {g:>12}  {str(r['attempts']):>3} {str(r['seconds']):>7}  {r['status']}")
    for t, e in errored:
        print(f"  ERROR {t}: {str(e)[:120]}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

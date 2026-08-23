"""nb_report.py — one markdown table describing the working NatureBench subset.

Merges, per task: tier (CPU/GPU24/GPU80), data size, setup cost (seconds,
overlay MB, overlay inodes), any setup-extra repair, the probe result (empty
submission -> official failure penalty), and the agent g from a named batch.

Usage:
    python nb_report.py --batch nb9b_v12 --probe-batch probe12 --out SUBSET.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NB_ROOT = Path(__file__).resolve().parent.parent


def load_g(batch_dir: Path, task: str):
    rj = batch_dir / task / "result.json"
    if not rj.exists():
        return None, "no-result"
    try:
        r = json.loads(rj.read_text())
    except Exception:
        return None, "bad-json"
    return r.get("g"), r.get("status", "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None, help="agent batch name")
    ap.add_argument("--probe-batch", default=None)
    ap.add_argument("--results-dir", default=str(NB_ROOT / "results"))
    ap.add_argument("--tasks-file", default=str(NB_ROOT / "task-sets" / "working.txt"))
    ap.add_argument("--out", default=str(NB_ROOT / "SUBSET.md"))
    args = ap.parse_args()

    setup = json.loads((NB_ROOT / "task-sets" / "setup_report.json").read_text())
    sizes = {e["task"]: e for e in json.loads((NB_ROOT / "task_sizes.json").read_text())}
    tiers = {}
    for name, f in (("CPU", "cpu.txt"), ("GPU24", "gpu_low.txt"), ("GPU80", "gpu_high.txt")):
        p = NB_ROOT / "repo" / "task-set" / f
        if p.exists():
            for line in p.read_text().split():
                tiers[line.strip()] = name
    tasks = [l.strip() for l in open(args.tasks_file) if l.strip() and not l.startswith("#")]
    res_dir = Path(args.results_dir)

    rows = []
    for t in tasks:
        s = setup.get(t, {})
        probe_g = probe_st = agent_g = agent_st = None
        if args.probe_batch:
            probe_g, probe_st = load_g(res_dir / args.probe_batch, t)
        if args.batch:
            agent_g, agent_st = load_g(res_dir / args.batch, t)
        rows.append({
            "task": t, "tier": tiers.get(t, "?"),
            "data_mb": s.get("data_mb"), "files": sizes.get(t, {}).get("files"),
            "setup_s": s.get("setup_seconds"), "overlay_mb": s.get("overlay_mb"),
            "overlay_inodes": s.get("overlay_inodes"),
            "repair": "yes" if s.get("setup_extra") else "",
            "probe_g": probe_g, "agent_g": agent_g, "agent_status": agent_st,
        })

    def fmt(v, nd=4):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    lines = ["| task | tier | data MB | setup s | overlay MB | overlay inodes | repair | probe g | agent g |",
             "|---|---|---:|---:|---:|---:|:--:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['tier']} | {fmt(r['data_mb'],1)} | {fmt(r['setup_s'],1)} | "
            f"{fmt(r['overlay_mb'],1)} | {fmt(r['overlay_inodes'])} | {r['repair']} | "
            f"{fmt(r['probe_g'])} | {fmt(r['agent_g'])} |")

    tot_ovl = sum(r["overlay_mb"] or 0 for r in rows)
    tot_ino = sum(r["overlay_inodes"] or 0 for r in rows)
    tot_data = sum(r["data_mb"] or 0 for r in rows)
    tot_setup = sum(r["setup_s"] or 0 for r in rows)
    lines.append("")
    lines.append(f"**Totals for {len(rows)} tasks**: data {tot_data/1000:.2f} GB, "
                 f"overlays {tot_ovl/1000:.2f} GB / {tot_ino} inodes "
                 f"({tot_ino//max(len(rows),1)} per task), "
                 f"total login-node setup {tot_setup/60:.1f} min.")

    gs = [r["agent_g"] for r in rows if isinstance(r["agent_g"], (int, float))]
    if gs:
        n = len(rows)
        lines.append(f"**Agent batch `{args.batch}`**: scored {len(gs)}/{n}; "
                     f"Match-SOTA (g>=0) {100*sum(g>=0 for g in gs)/n:.1f}%, "
                     f"Surpass-SOTA (g>0.1) {100*sum(g>0.1 for g in gs)/n:.1f}%, "
                     f"mean g {sum(gs)/len(gs):.4f}, "
                     f"distinct values {len(set(round(g,6) for g in gs))}.")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

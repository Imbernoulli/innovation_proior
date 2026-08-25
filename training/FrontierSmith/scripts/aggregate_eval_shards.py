#!/usr/bin/env python3
"""Combine shard_*/summary_shard.json of a split eval into ONE comparable row.

Why this exists: the split-eval writes one summary per shard, and every shard
metric is a *per-problem mean over that shard's problems only*. Eyeballing two
shards, or averaging them blindly, silently changes the denominator between
arms -- which is exactly the mistake that makes an arm look better than base.
Here the combination is always a problem-count-weighted mean, and the counts
are printed so an unequal split is visible instead of hidden.

Denominators: FCS count comes from official_leaderboard_metrics.frontiercs
.num_problems; the ALE count is complete_problem_count minus that.

  usage: aggregate_eval_shards.py <eval_dir> [<eval_dir> ...]
         aggregate_eval_shards.py --glob 'outputs/cc_eval_ja_v2_*'
"""
import json, sys, glob, pathlib

FCS_FIELDS = ["avg_at_5", "score_at_1", "score_at_5", "pass_at_1", "pass_at_5"]
ALE_FIELDS = [("performance", "mean@5"), ("performance", "best@5/mean")]


def load(d):
    rows = []
    for p in sorted(glob.glob(str(pathlib.Path(d) / "shard_*" / "summary_shard.json"))):
        try:
            rows.append(json.load(open(p)))
        except Exception as e:
            print(f"  !! unreadable {p}: {e}", file=sys.stderr)
    return rows


def combine(d):
    shards = load(d)
    if not shards:
        return None
    fcs_n = ale_n = 0
    missing_ale = [0]
    fcs = {k: 0.0 for k in FCS_FIELDS}
    ale = {f"{a}.{b}": 0.0 for a, b in ALE_FIELDS}
    for s in shards:
        lb = (s.get("official_leaderboard_metrics") or {}).get("frontiercs") or {}
        n = lb.get("num_problems") or 0
        if n:
            fcs_n += n
            for k in FCS_FIELDS:
                fcs[k] += (lb.get(k) or 0.0) * n
        na = (s.get("complete_problem_count") or 0) - n
        am = (s.get("metrics") or {}).get("alebench") or {}
        # A shard whose ALE never scored writes None, not 0. Folding None into
        # 0.0 turns "we have no ALE number" into "ALE scored zero" -- which is
        # how the agentic-ablation table showed soup_noag_a10 at ALE 0.0 when
        # in truth its ALE was never computed (the dockerd outage). Count only
        # shards that actually produced a value, and report the shortfall.
        if na > 0:
            vals = {f"{a}.{b}": (am.get(a) or {}).get(b) for a, b in ALE_FIELDS}
            if all(v is not None for v in vals.values()):
                ale_n += na
                for k, v in vals.items():
                    ale[k] += v * na
            else:
                missing_ale[0] += na
    out = {"name": pathlib.Path(d).name, "shards": len(shards),
           "fcs_problems": fcs_n,
           "ale_problems": (f"{ale_n}(+{missing_ale[0]}?)" if missing_ale[0] else ale_n)}
    for k in FCS_FIELDS:
        out["fcs_" + k] = round(fcs[k] / fcs_n, 4) if fcs_n else None
    for a, b in ALE_FIELDS:
        key = f"{a}.{b}"
        out["ale_" + key] = round(ale[key] / ale_n, 2) if ale_n else "NOT-SCORED"
    return out


def main(argv):
    dirs = []
    if argv and argv[0] == "--glob":
        for pat in argv[1:]:
            dirs += sorted(glob.glob(pat))
    else:
        dirs = argv
    rows = [r for r in (combine(d) for d in dirs) if r]
    if not rows:
        print("no shard summaries found", file=sys.stderr); return 1
    hdr = ["name", "shards", "fcs_problems", "fcs_avg_at_5", "fcs_score_at_1",
           "fcs_pass_at_1", "fcs_pass_at_5", "ale_problems",
           "ale_performance.mean@5", "ale_performance.best@5/mean"]
    w = [max(len(h), *(len(str(r.get(h))) for r in rows)) for h in hdr]
    print("  ".join(h.ljust(x) for h, x in zip(hdr, w)))
    for r in rows:
        print("  ".join(str(r.get(h)).ljust(x) for h, x in zip(hdr, w)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

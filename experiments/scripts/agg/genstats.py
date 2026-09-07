#!/usr/bin/env python3
"""genstats.py -- statistics for the generation-side research-taste tasks.

Two families, two treatments:

  numeric (openreview_score, openreview_novel)
      The model predicts a 0-10 rating; the target is the real reviewer mean. We report
      Spearman rho, and compare two arms with a bootstrap that resamples ITEMS, not
      draws. Resampling draws would treat the 4 samples of one paper as 4 independent
      observations -- they are not, and the earlier campaign already showed that
      sample-level bootstrap overstates confidence (0.890 -> 0.951 on the same contrast
      once it was done at problem level).

  gen (liveidea_gen, review_weakness)
      No key exists, so there is nothing to score here. judge_pairwise.py settles those;
      this file only reports coverage so a half-finished run is visible.

WHY THE PAIRED BOOTSTRAP RESAMPLES ITEMS TOGETHER FOR BOTH ARMS
Both arms answered the SAME papers. Drawing the item set once and scoring both arms on
it cancels the item-difficulty variance, which is the dominant term. Drawing
independently for each arm would leave it in and hide a real difference behind noise.

REPORTING P(>0)
The printed P is the fraction of bootstrap replicates above zero. It is NOT a p-value
and it is not a probability that the effect is real -- Codex flagged the earlier report
for writing P=1.000 as certainty. 1.000 here means "no non-positive replicate in 10000
draws", nothing stronger, and it says nothing about training-seed variance because we
have one seed per arm.

  python3 genstats.py table openreview_score tagA tagB ...
  python3 genstats.py pair  openreview_score ourtag controltag
"""
import argparse, glob, json, os, random, sys
from collections import defaultdict

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"


def load(tag, outroot=None):
    """tag -> {task: {item_id: (pred_mean, target)}} plus gen coverage."""
    root = outroot or f"{D}/outputs"
    p = f"{root}/cc_gen_{tag}/samples.jsonl"
    if not os.path.exists(p):
        return None
    num = defaultdict(lambda: defaultdict(list))
    tgt = defaultdict(dict)
    gen = defaultdict(lambda: defaultdict(int))
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        if r.get("kind") == "numeric":
            if r.get("pred") is not None:
                num[r["task"]][r["id"]].append(float(r["pred"]))
            if r.get("target") is not None:
                tgt[r["task"]][r["id"]] = float(r["target"])
        elif (r.get("text") or "").strip():
            gen[r["task"]][r["id"]] += 1
    out = {}
    for task, items in num.items():
        out[task] = {i: (sum(v) / len(v), tgt[task][i])
                     for i, v in items.items() if i in tgt[task] and v}
    return out, {t: dict(v) for t, v in gen.items()}


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def common_items(datas, task):
    sets = [set(d[0].get(task, {})) for d in datas]
    if not sets or not all(sets):
        return []
    return sorted(set.intersection(*sets))


def cmd_table(a):
    datas, tags = [], []
    for t in a.tags:
        d = load(t, a.outroot)
        if d is None:
            print(f"  (no output for {t})", file=sys.stderr)
            continue
        datas.append(d)
        tags.append(t)
    if not datas:
        sys.exit("no arms with output")
    items = common_items(datas, a.task)
    print(f"\n{a.task}: {len(items)} items common to all {len(tags)} arms\n")
    print(f"  {'arm':<30s} {'spearman':>9s} {'n':>5s} {'pred_mean':>10s} {'pred_sd':>8s}")
    rows = []
    for t, d in zip(tags, datas):
        m = d[0].get(a.task, {})
        xs = [m[i][0] for i in items]
        ys = [m[i][1] for i in items]
        rho = spearman(xs, ys)
        mu = sum(xs) / len(xs) if xs else 0.0
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
        rows.append((rho if rho is not None else -9, t, rho, len(xs), mu, sd))
    for _, t, rho, n, mu, sd in sorted(rows, reverse=True):
        rs = f"{rho:+.4f}" if rho is not None else "   n/a"
        print(f"  {t:<30s} {rs:>9s} {n:>5d} {mu:>10.2f} {sd:>8.2f}")
    print()


def cmd_pair(a):
    da, db = load(a.ours, a.outroot), load(a.control, a.outroot)
    if da is None or db is None:
        sys.exit(f"missing output for {a.ours if da is None else a.control}")
    items = common_items([da, db], a.task)
    if len(items) < 10:
        sys.exit(f"only {len(items)} common items for {a.task}")
    ma, mb = da[0][a.task], db[0][a.task]
    xa = [ma[i][0] for i in items]
    xb = [mb[i][0] for i in items]
    y = [ma[i][1] for i in items]

    ra, rb = spearman(xa, y), spearman(xb, y)
    rng = random.Random(20260907)
    n = len(items)
    diffs = []
    for _ in range(a.n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = spearman([xa[i] for i in idx], [y[i] for i in idx])
        sb = spearman([xb[i] for i in idx], [y[i] for i in idx])
        if sa is not None and sb is not None:
            diffs.append(sa - sb)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    p = sum(1 for d in diffs if d > 0) / len(diffs)

    print(f"\n{a.task}  n={n} common items   [{a.n_boot} item-level bootstrap replicates]")
    print(f"  {a.ours:<30s} spearman {ra:+.4f}")
    print(f"  {a.control:<30s} spearman {rb:+.4f}")
    print(f"  diff = {ra - rb:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   P(>0) = {p:.3f}")
    if p >= 0.999:
        print("  (P is the share of replicates above zero, not a p-value; it does not")
        print("   cover training-seed variance -- there is one seed per arm.)")
    print()


def cmd_coverage(a):
    for t in a.tags:
        d = load(t, a.outroot)
        if d is None:
            print(f"  {t:<30s} NO OUTPUT")
            continue
        num, gen = d
        bits = []
        for task in sorted(num):
            bits.append(f"{task}={len(num[task])}")
        for task in sorted(gen):
            full = sum(1 for v in gen[task].values() if v >= a.n_samples)
            bits.append(f"{task}={len(gen[task])}({full} full)")
        print(f"  {t:<30s} " + "  ".join(bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-samples", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("table"); t.add_argument("task"); t.add_argument("tags", nargs="+")
    p = sub.add_parser("pair"); p.add_argument("task")
    p.add_argument("ours"); p.add_argument("control")
    c = sub.add_parser("coverage"); c.add_argument("tags", nargs="+")

    a = ap.parse_args()
    {"table": cmd_table, "pair": cmd_pair, "coverage": cmd_coverage}[a.cmd](a)


if __name__ == "__main__":
    main()

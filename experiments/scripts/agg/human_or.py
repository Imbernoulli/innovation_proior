"""Human references for the OpenReview-derived tasks in EXTRA_BENCH.

Three separate references, each computed on exactly the items the benchmark uses:

  openreview_score / openreview_novel -- a single human referee is the ceiling.
      For every paper we hold out one reviewer and correlate that reviewer's own
      score against the mean of the remaining reviewers, then Spearman across
      papers. That is the same quantity our models are asked for (predict the
      panel mean) with a human doing the predicting -- except the human read the
      whole paper and our models see title + abstract, so it is an upper bound,
      not a like-for-like contest.

  openreview_decide -- the reviewers' own verdict. Threshold the panel's mean
      score and predict accept/reject. Reported at the threshold that maximises
      accuracy (an oracle threshold, so again an upper bound) and at the median.

  openreview_pair -- nothing to compute: the label IS which paper the reviewers
      scored higher, so a human referee panel is 100% by construction.
"""
import json, glob, os, statistics, re, sys
import pandas as pd

IB = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/ideabench"


def spearman(x, y):
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
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def load_df():
    return pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(f"{IB}/openreview/data/*.parquet"))], ignore_index=True)


def titles(task, path):
    out = []
    for l in open(path):
        r = json.loads(l)
        if r["task"] != task:
            continue
        m = re.search(r"^Title: (.+)$", r["prompt"], re.M)
        if m:
            out.append(m.group(1).strip())
    return out


def loo(df, want, field):
    """Leave-one-reviewer-out: one referee's score vs the mean of the others."""
    idx = {str(t).strip(): i for i, t in enumerate(df["title"])}
    one, rest, npapers, nrev = [], [], 0, []
    for t in want:
        i = idx.get(t)
        if i is None:
            continue
        revs = df.iloc[i]["reviews"]
        vals = [r[field] for r in revs if r.get(field) is not None] if revs is not None else []
        if len(vals) < 2:
            continue
        npapers += 1
        nrev.append(len(vals))
        for k in range(len(vals)):
            others = vals[:k] + vals[k + 1:]
            one.append(vals[k]); rest.append(sum(others) / len(others))
    return spearman(one, rest), npapers, (statistics.median(nrev) if nrev else 0), len(one)


def decide(df, want):
    idx = {str(t).strip(): i for i, t in enumerate(df["title"])}
    rows = []
    for t in want:
        i = idx.get(t)
        if i is None:
            continue
        r = df.iloc[i]
        if r["mean_score"] is None or pd.isna(r["mean_score"]) or r["decision"] is None:
            continue
        rows.append((float(r["mean_score"]), bool(r["decision"])))
    if not rows:
        return None
    best = max(((sum((s >= th) == d for s, d in rows) / len(rows), th)
                for th in sorted({s for s, _ in rows})), key=lambda x: x[0])
    med = statistics.median(s for s, _ in rows)
    at_med = sum((s >= med) == d for s, d in rows) / len(rows)
    return len(rows), best[0] * 100, best[1], at_med * 100, sum(d for _, d in rows)


def main():
    df = load_df()
    print(f"OpenReview 语料:{len(df)} 篇,带 `reviews` 逐条评审分。\n")
    for src, label in [(f"{IB}/gentasks.jsonl", "gentasks(§3 用的那 120 篇)")]:
        for task, field in [("openreview_score", "score"), ("openreview_novel", "novelty")]:
            want = titles(task, src)
            rho, npap, medrev, npairs = loo(df, want, field)
            print(f"- `{task}` {label}:题目 {len(want)} 篇,匹配上 {npap} 篇,"
                  f"每篇中位 {medrev} 条评审。**留一评审 ρ = {rho:+.4f}**({npairs} 个留一对)")
    print()
    for src, label in [(f"{IB}/tasks_v2.jsonl", "tasks_v2(§2.3 的 120 题)"),
                       (f"{IB}/tasks.jsonl", "tasks(旧版 120 题)")]:
        want = titles("openreview_decide", src)
        d = decide(df, want)
        if d:
            n, acc, th, accmed, npos = d
            print(f"- `openreview_decide` {label}:匹配上 {n}/{len(want)} 篇(其中 {npos} 篇录用)。"
                  f"**用评审均分预测录用:最佳阈值 {th:.3f} 下 {acc:.2f}%;中位阈值下 {accmed:.2f}%**")


main()

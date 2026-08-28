#!/usr/bin/env python3
"""One table for the 9B family: every bench, every arm, paired against base, Holm-corrected.

Item-paired throughout — each arm is compared to base on the SAME items, so the
comparison is not contaminated by which items each arm happened to answer. Holm
runs over the whole family of arm x bench tests at once, which is the number that
answers "where are we significantly better".
"""
import json, os, sys, math, random, collections, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benches import extract_ab, extract_bucket, extract_rating_1_5, extract_letter

R9 = "outputs_taste/run9b"
JU = "outputs_taste/judged"
ARMS = ["base", "soup_a10", "soup_wd03_a20", "rl_base", "rl_soupNEW10", "rl_soupWD03"]


def _rows(path):
    if not os.path.exists(path):
        return None
    d = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            d[r["id"]] = r
    return d


def per_item(bench, arm):
    """id -> score, higher is better. None when the arm's file is missing."""
    if bench == "giants":
        rows = _rows(f"{JU}/judged_q9b_giants_{arm}.jsonl")
        if rows is None:
            return None
        # same convention as score.py:score_giants -- an item the model never
        # produced an insight for is FLOORED to the rubric minimum rather than
        # dropped, because dropping it would reward whichever arm truncates most.
        # Judge-infrastructure errors are excluded: those are ours, not the model's.
        out = {}
        for i, r in rows.items():
            v = r.get("rating")
            if v is None:
                if r.get("reason") == "no_insight":
                    out[i] = 1.0
                continue
            out[i] = float(v)
        return out
    rows = _rows(f"{R9}/cc_eval_q9b_{arm}_{bench}.jsonl")
    if rows is None:
        return None
    out = {}
    if bench in ("scijudge_nothink", "pairiq"):
        pr = collections.defaultdict(dict)
        for i, r in rows.items():
            pr[r["meta"]["pair"]]["swap" if r["meta"]["swap"] else "plain"] = r
        for pid, d in pr.items():
            if len(d) < 2:
                continue
            out[pid] = float(all(extract_ab(d[k]["output"] or "") == d[k]["meta"]["gold"]
                                 for k in ("plain", "swap")))
    elif bench == "soundness":
        for i, r in rows.items():
            b, _ = extract_bucket(r["output"] or "")
            out[i] = float(b == r["meta"]["gold"])
    elif bench == "rino":
        for i, r in rows.items():
            v = extract_rating_1_5(r["output"] or "")
            out[i] = -abs((v if v else 3) - r["meta"]["gold"])
    elif bench == "scipredict":
        for i, r in rows.items():
            out[i] = float(extract_letter(r["output"] or "") == r["meta"]["gold"])
    return out


def paired(a, b, n=8000, seed=7):
    ks = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in ks]
    if not d:
        return None
    m = sum(d) / len(d)
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n))
    # two-sided p from the bootstrap sign
    p = 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / n
    return m, bs[int(.025 * n)], bs[int(.975 * n)], min(1.0, p), len(ks)


MIN_COVERAGE = 0.9   # an arm judged on <90% of base's items is not comparable yet


def main():
    benches = [("scijudge_nothink", "换序一致率"), ("pairiq", "换序一致率"),
               ("soundness", "准确率"), ("scipredict", "准确率"),
               ("rino", "负MAE"), ("giants", "洞察相似度1-10")]
    tests, incomplete = [], []
    print(f"{'bench':20s}{'指标':14s}" + "".join(f"{a[:12]:>14s}" for a in ARMS))
    for b, label in benches:
        D = {a: per_item(b, a) for a in ARMS}
        if not D.get("base"):
            print(f"{b:20s}{'(base 缺)':14s}")
            continue
        # Refuse partially-judged arms. A judging run still in flight yields a
        # short file that scores perfectly happily and can invent a large fake
        # effect -- rl_soupNEW10 read -0.803 on giants at 79 of 400 items.
        nbase = len(D["base"])
        for a in list(D):
            if D[a] is not None and len(D[a]) < MIN_COVERAGE * nbase:
                incomplete.append((b, a, len(D[a]), nbase))
                D[a] = None
        line = f"{b:20s}{label:14s}"
        for a in ARMS:
            line += f"{(sum(D[a].values())/len(D[a])):14.3f}" if D[a] else f"{'—':>14s}"
        print(line)
        line2 = f"{'':20s}{'Δ vs base':14s}"
        for a in ARMS:
            if a == "base" or not D[a]:
                line2 += f"{'':>14s}"
                continue
            r = paired(D["base"], D[a])
            if r is None:
                line2 += f"{'':>14s}"
                continue
            m, lo, hi, p, n = r
            tests.append((b, a, m, lo, hi, p, n))
            line2 += f"{m:+11.3f}   "
        print(line2)

    if incomplete:
        print("\n未完成、已从对照中剔除（判分仍在进行）:")
        for b, a, n, nb in incomplete:
            print(f"  {b:20s}{a:16s}{n:5d}/{nb} 项 = {n/nb*100:.0f}%")

    if not tests:
        return
    tests.sort(key=lambda t: t[5])
    k = len(tests)
    print(f"\nHolm 校正（family = {k} 个 臂×bench 对照，alpha=0.05）")
    print(f"{'bench':20s}{'arm':16s}{'Δ':>9}{'95%CI':>22}{'p':>9}{'阈值':>9}  结论")
    rejected = True
    for i, (b, a, m, lo, hi, p, n) in enumerate(tests):
        thr = 0.05 / (k - i)
        if p > thr:
            rejected = False
        sig = "显著" if (rejected and p <= thr) else ""
        print(f"{b:20s}{a:16s}{m:+9.3f}  [{lo:+.3f},{hi:+.3f}]{p:9.4f}{thr:9.4f}  {sig}")
    n_sig = sum(1 for i, t in enumerate(tests) if t[5] <= 0.05 / (k - i))
    print(f"\n未校正 p<0.05 的对照: {sum(1 for t in tests if t[5]<0.05)}/{k}")


if __name__ == "__main__":
    main()

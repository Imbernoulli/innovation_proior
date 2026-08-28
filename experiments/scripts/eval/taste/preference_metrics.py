#!/usr/bin/env python3
"""Calibration-free re-scoring of the rating benchmarks.

Absolute accuracy on a 1-5 or high/low label confounds two different things:
whether a model can ORDER items correctly, and where it happens to put its
scale.  Our arms sit at very different places on the scale (RINoBench mean
prediction: human gold 3.21, base 3.57, wd01 3.95), so exact-match accuracy and
macro-F1 partly measure the offset, not the discrimination.

This reports the offset-free versions:
  RINoBench  -> pairwise rank concordance over gold-differing pairs (ties 0.5)
  Soundness  -> AUC over an ordinal score built from bucket x confidence
Both have chance = 0.5 and are invariant to any monotone rescaling of the
model's own scale.
"""
from __future__ import annotations
import itertools, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benches import extract_bucket, extract_rating_1_5  # noqa: E402


def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        d[r["id"]] = r
    return list(d.values())


def auc_fast(S, L):
    order = sorted(range(len(S)), key=lambda i: S[i])
    ranks = [0.0] * len(S)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and S[order[j + 1]] == S[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    npos = sum(L); nneg = len(L) - npos
    if not npos or not nneg:
        return None
    return (sum(r for r, y in zip(ranks, L) if y == 1) - npos * (npos + 1) / 2) / (npos * nneg)


def boot_auc(S, L, n=1000, seed=7):
    rng = random.Random(seed); m = len(S); out = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        a = auc_fast([S[i] for i in idx], [L[i] for i in idx])
        if a is not None:
            out.append(a)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def concordance(pairs):
    c = d = t = 0
    for (p1, g1), (p2, g2) in itertools.combinations(pairs, 2):
        if g1 == g2:
            continue
        if p1 == p2:
            t += 1
        elif (p1 > p2) == (g1 > g2):
            c += 1
        else:
            d += 1
    n = c + d + t
    return ((c + 0.5 * t) / n if n else float("nan")), n


FAM = [("4B", "outputs_taste/run4b_redo", "cc_eval_pp_{a}_{b}.jsonl",
        ["base", "wd01", "soup_a10"]),
       ("9B", "outputs_taste/run9b", "cc_eval_q9b_{a}_{b}.jsonl",
        ["base", "soup_a10", "soup_wd03_a20", "rl_base", "rl_soupNEW10", "rl_soupWD03"])]

if __name__ == "__main__":
    print("=== RINoBench pairwise rank concordance (offset-free, chance 0.5)")
    for fam, d, tpl, arms in FAM:
        for a in arms:
            f = os.path.join(d, tpl.format(a=a, b="rino"))
            if not os.path.exists(f):
                continue
            pr = [(extract_rating_1_5(r["output"] or ""), r["meta"]["gold"]) for r in load(f)]
            pr = [(p, g) for p, g in pr if p is not None]
            v, n = concordance(pr)
            print(f"  {fam} {a:15s} {v:.4f}   (n_pairs={n})")
    print("\n=== SoundnessBench AUC (bucket x confidence, threshold-free, chance 0.5)")
    for fam, d, tpl, arms in FAM:
        for a in arms:
            f = os.path.join(d, tpl.format(a=a, b="soundness"))
            if not os.path.exists(f):
                continue
            S, L = [], []
            for r in load(f):
                b, c = extract_bucket(r["output"] or "")
                if b is None:
                    continue
                c = c or 3
                S.append(c if b == "high" else -c)
                L.append(1 if r["meta"]["gold"] == "high" else 0)
            A = auc_fast(S, L); lo, hi = boot_auc(S, L)
            print(f"  {fam} {a:15s} AUC={A:.4f}  95%CI[{lo:.4f},{hi:.4f}]  n={len(S)}")

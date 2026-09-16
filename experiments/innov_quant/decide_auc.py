#!/usr/bin/env python3
"""openreview_decide: separate DISCRIMINATION from DECISION THRESHOLD.

Plain accuracy says RL makes research judgment worse. But the gold set is exactly 50/50,
and the per-arm breakdown shows accuracy on gold=ACCEPT is flat while accuracy on
gold=REJECT collapses: the model got more permissive, which costs 0/1 accuracy on a
balanced set even if its ability to rank papers is untouched.

Accuracy cannot tell those apart. Each item is answered 5 times, so the fraction of draws
that say ACCEPT is a graded confidence in [0,1]; the AUC of that score against the gold
label is threshold-free and measures ranking ability alone. Reporting both, plus the
accept rate, says exactly which of the two moved.

CI on a paired AUC difference by bootstrapping ITEMS (both arms resampled together, so
the pairing is kept); ties in the score contribute 0.5 as usual.
"""
import json, collections, os
import numpy as np

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
FAM = {"idea": "cc_idea32k_{}_y26pp", "ideav2": "cc_ideav2_{}_y26pp"}
ARMS = [("9B base", "base9b_v2c"), ("9B SFT", "ft01mix_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL(SFT)", "rlv5_ft01mix_a10_s20"),
        ("4B base", "base4b"), ("4B SFT", "4b_ft01mix_a10"),
        ("4B RL(base)", "rlv5_4b_base_s20"), ("4B RL(SFT)", "rlv5_4b_ft01mix_a10_s20")]
TASK = "openreview_decide"


def load(fam, arm):
    p = os.path.join(D, FAM[fam].format(arm), "samples.jsonl")
    if not os.path.exists(p):
        return None
    acc = collections.defaultdict(list)
    gold = {}
    for ln in open(p):
        r = json.loads(ln)
        if r.get("task") != TASK:
            continue
        i = str(r.get("id"))
        gold[i] = 1 if r.get("answer") == "ACCEPT" else 0
        # unparsed draws carry no decision; drop them rather than scoring them as REJECT
        if r.get("unparsed"):
            continue
        acc[i].append(1.0 if str(r.get("pred")) == "ACCEPT" else 0.0)
    return {i: float(np.mean(v)) for i, v in acc.items() if v}, gold


def auc(score, y):
    """Mann-Whitney AUC with ties at 0.5."""
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / (len(pos) * len(neg)))


def main():
    rng = np.random.default_rng(0)
    for fam in FAM:
        cache = {n: load(fam, a) for n, a in ARMS}
        print(f"\n===== {fam} / {TASK}")
        print(f"{'arm':14s} {'items':>5s} {'acc':>7s} {'acceptR':>8s} {'AUC':>7s}  {'AUC 95% CI':>16s}")
        ids0 = None
        for n, _ in ARMS:
            if cache[n] is None:
                print(f"{n:14s} MISSING")
                continue
            s, g = cache[n]
            ids = sorted(set(s) & set(g))
            ids0 = ids if ids0 is None else ids0
            sc = np.array([s[i] for i in ids])
            y = np.array([g[i] for i in ids])
            a = auc(sc, y)
            bs = []
            for _ in range(2000):
                k = rng.integers(0, len(ids), len(ids))
                v = auc(sc[k], y[k])
                if np.isfinite(v):
                    bs.append(v)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            accu = float(np.mean(np.where(y == 1, sc, 1 - sc)))
            print(f"{n:14s} {len(ids):5d} {accu:7.3f} {sc.mean():8.3f} {a:7.3f}  [{lo:6.3f},{hi:6.3f}]")

        # paired AUC differences on the common item set
        print(f"  {'contrast':34s} {'dAUC':>7s}  {'95% CI':>16s}  {'dAcc':>7s}")
        for A, B in [("9B RL(SFT)", "9B base"), ("9B RL(SFT)", "9B RL(base)"), ("9B SFT", "9B base"),
                     ("4B RL(SFT)", "4B base"), ("4B RL(SFT)", "4B RL(base)"), ("4B SFT", "4B base")]:
            if cache[A] is None or cache[B] is None:
                continue
            sa, ga = cache[A]
            sb, _ = cache[B]
            ids = sorted(set(sa) & set(sb) & set(ga))
            xa = np.array([sa[i] for i in ids])
            xb = np.array([sb[i] for i in ids])
            y = np.array([ga[i] for i in ids])
            d = auc(xa, y) - auc(xb, y)
            da = float(np.mean(np.where(y == 1, xa, 1 - xa)) - np.mean(np.where(y == 1, xb, 1 - xb)))
            bs = []
            for _ in range(2000):
                k = rng.integers(0, len(ids), len(ids))
                v = auc(xa[k], y[k]) - auc(xb[k], y[k])
                if np.isfinite(v):
                    bs.append(v)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {A+' - '+B:34s} {d:+7.3f}  [{lo:+6.3f},{hi:+6.3f}]  {da:+7.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""openreview_decide: the base model's lead is venue-name recall, not research judgment.

Plain accuracy on this task says RL makes judgment worse (9B RL(SFT) - base = -0.081).
Two things had to be ruled out before believing that.

1. THRESHOLD vs DISCRIMINATION. The gold set is exactly 50/50, and the per-arm split
   shows accuracy on gold=ACCEPT flat while accuracy on gold=REJECT collapses: the arms
   got more permissive. Accuracy cannot separate a moved threshold from lost ranking
   ability, so score each item by the fraction of its 5 draws saying ACCEPT and take the
   AUC, which is threshold-free. That did NOT rescue the result (-0.115 AUC), so the
   threshold story is refuted -- discrimination really is lower with the venue in the
   prompt.

2. THE VENUE SHORTCUT. idea_prep.py's v1 prompt names the venue ("submitted to ICLR
   2023"). Venue acceptance rates differ a lot, so an arm can score by recalling a base
   rate without reading the paper. The v1 note argued this "does not bias the RANKING
   because the shortcut is open to every arm equally" -- that argument only holds if
   every arm retains the memorised base rates equally, and RL is exactly the kind of
   training that would not.

   --venue-ablation is the controlled test and had already been run: same papers, same
   ids, same labels, the venue sentence removed and nothing else. Comparing each arm
   with and without it isolates how much of that arm's score was the shortcut.

CI by bootstrapping ITEMS (arms resampled together, so pairing is kept). Ties 0.5.
Caveat kept in the output: the ablation is an independent generation run, so per-arm AUC
also carries sampling noise that an item bootstrap does not see.
"""
import json, collections, os
import numpy as np

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
ARMS = [("9B base", "base9b_v2c"), ("9B SFT", "ft01mix_a10"),
        ("9B RL(base)", "rlv5_base_s20"), ("9B RL(SFT)", "rlv5_ft01mix_a10_s20")]
TASK = "openreview_decide"


def load(d):
    p = os.path.join(D, d, "samples.jsonl")
    if not os.path.exists(p):
        return None
    sc, gold, unp, n = collections.defaultdict(list), {}, 0, 0
    for ln in open(p):
        r = json.loads(ln)
        if r.get("task") != TASK:
            continue
        n += 1
        i = str(r.get("id"))
        gold[i] = 1 if r.get("answer") == "ACCEPT" else 0
        if r.get("unparsed"):
            unp += 1
            continue
        sc[i].append(1.0 if str(r.get("pred")) == "ACCEPT" else 0.0)
    return {i: float(np.mean(v)) for i, v in sc.items() if v}, gold, n, unp / max(n, 1)


def auc(sc, gold, ids):
    pos = [i for i in ids if gold[i] == 1]
    neg = [i for i in ids if gold[i] == 0]
    if not pos or not neg:
        return float("nan")
    t = sum((1.0 if sc[p] > sc[q] else 0.5 if sc[p] == sc[q] else 0.0) for p in pos for q in neg)
    return t / (len(pos) * len(neg))


def boot(fn, ids, rng, n=3000):
    v = [fn([ids[j] for j in rng.integers(0, len(ids), len(ids))]) for _ in range(n)]
    v = [x for x in v if np.isfinite(x)]
    return np.percentile(v, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    W = {n: load(f"cc_idea32k_{a}_y26pp") for n, a in ARMS}
    O = {n: load(f"cc_venueabl_{a}") for n, a in ARMS}
    names = [n for n, _ in ARMS if W[n] and O[n]]
    ids = sorted(set.intersection(*[set(W[n][0]) & set(O[n][0]) for n in names]))
    gold = W[names[0]][1]
    for n in names:                      # the ablation must be the same papers
        assert all(W[n][1][i] == O[n][1][i] for i in ids), f"gold labels differ for {n}"

    print(f"{TASK}: 受控消融 (同一批论文/id/标签, 只去掉 prompt 里的会议名), n={len(ids)} 题")
    print(f"{'arm':12s} {'AUC 有会议':>10s} {'AUC 无会议':>10s} {'掉幅':>8s} {'掉幅 95% CI':>18s} {'unparsed':>9s}")
    A = {}
    for n in names:
        sw, so = W[n][0], O[n][0]
        aw, ao = auc(sw, gold, ids), auc(so, gold, ids)
        A[n] = (aw, ao)
        lo, hi = boot(lambda s: auc(sw, gold, s) - auc(so, gold, s), ids, rng)
        print(f"{n:12s} {aw:10.3f} {ao:10.3f} {aw-ao:+8.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"{W[n][3]:>4.1%}/{O[n][3]:<4.1%}")

    print(f"\n{'vs 9B base':22s} {'dAUC 有会议':>11s} {'95% CI':>18s} | {'dAUC 无会议':>11s} {'95% CI':>18s}")
    b = "9B base"
    for n in names:
        if n == b:
            continue
        sa, sb = W[n][0], W[b][0]
        oa, ob = O[n][0], O[b][0]
        dw, do = auc(sa, gold, ids) - auc(sb, gold, ids), auc(oa, gold, ids) - auc(ob, gold, ids)
        lw = boot(lambda s: auc(sa, gold, s) - auc(sb, gold, s), ids, rng)
        lo_ = boot(lambda s: auc(oa, gold, s) - auc(ob, gold, s), ids, rng)
        print(f"{n:22s} {dw:+11.3f} [{lw[0]:+7.3f},{lw[1]:+7.3f}] | {do:+11.3f} [{lo_[0]:+7.3f},{lo_[1]:+7.3f}]")

    drops = [A[n][0] - A[n][1] for n in names]
    print(f"\n跨臂方向: 掉幅按 base > SFT > RL(base) > RL(SFT) 单调排列 = "
          f"{drops == sorted(drops, reverse=True)}  ({', '.join(f'{d:+.3f}' for d in drops)})")
    print("注意: 消融是一次独立的生成跑批, 逐臂 AUC 还带生成侧抽样噪声, 题级 bootstrap 看不到这一层。")


if __name__ == "__main__":
    main()

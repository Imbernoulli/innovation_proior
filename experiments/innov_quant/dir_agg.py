#!/usr/bin/env python3
"""Directional aggregation across cells.

Why this exists: reporting each (bench x arm-pair) cell against "does its own 95% CI
exclude 0" is the wrong test for effects that are small but consistent. A cell here is
39-172 problems; at that size an effect of the size we actually have will almost always
give a CI touching 0. The right question for a claim like "the prior helps" is whether
the cells AGREE, so this aggregates them: a sign test over cell directions and a
Stouffer combination of the per-cell signed Z. Per-cell numbers are still printed so
nothing is hidden -- the aggregate is added, not substituted.

The replicate arms (`<arm>_y2026`) are run through the identical machinery so the same
aggregate can be read off a contrast where nothing was changed.
"""
import csv, sys, collections, math
import numpy as np
from scipy.stats import wilcoxon, binomtest, norm

SRC = "metrics.csv"

def load():
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(SRC) as f:
        for r in csv.DictReader(f):
            b, a, p = r["bench"], r["arm"], r["problem"]
            try:
                si = int(r["sample_idx"])
            except Exception:
                continue
            rows[b][a][(p, si)] = r
    return rows

DATA = load()

def cell(bench, A, B, field="score"):
    """Paired per-problem contrast A - B. Returns dict or None."""
    da, db = DATA[bench].get(A), DATA[bench].get(B)
    if not da or not db:
        return None
    keys = set(da) & set(db)
    byp = collections.defaultdict(lambda: ([], []))
    for k in keys:
        p = k[0]
        for src, idx in ((da, 0), (db, 1)):
            v = src[k].get(field)
            if v in (None, ""):
                continue
            if isinstance(v, str) and v.strip() in ("True", "False"):
                v = 1.0 if v.strip() == "True" else 0.0
            try:
                byp[p][idx].append(float(v))
            except ValueError:
                return None
    xa, xb = [], []
    for p, (la, lb) in byp.items():
        if la and lb:
            xa.append(np.mean(la)); xb.append(np.mean(lb))
    if len(xa) < 8:
        return None
    xa, xb = np.array(xa), np.array(xb)
    d = xa - xb
    nz = d[d != 0]
    if len(nz) < 5:
        return None
    try:
        st, p_two = wilcoxon(nz)
    except Exception:
        return None
    # signed Z from the two-sided p, sign taken from the median of the nonzero diffs
    s = 1.0 if np.median(nz) > 0 else -1.0
    z = s * abs(norm.ppf(max(p_two, 1e-12) / 2))
    return dict(bench=bench, A=A, B=B, n=len(d), mean=float(d.mean()),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), p=float(p_two), z=float(z))

def aggregate(cells, label):
    cells = [c for c in cells if c]
    if not cells:
        print(f"\n### {label}: no usable cells"); return
    print(f"\n### {label}")
    print("| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |")
    print("|---|---|---|---|---|---|")
    for c in cells:
        print(f"| {c['bench']} | {c['A']} − {c['B']} | {c['n']} | {c['mean']:+.3f} | {c['pos']}/{c['neg']} | {c['p']:.4f} |")
    npos = sum(1 for c in cells if c["mean"] > 0)
    k, n = npos, len(cells)
    sgn = binomtest(k, n, 0.5, alternative="two-sided").pvalue
    Z = sum(c["z"] for c in cells) / math.sqrt(len(cells))
    p_st = 2 * (1 - norm.cdf(abs(Z)))
    print(f"\n**聚合:{k}/{n} 个格子方向为正,符号检验 p={sgn:.4f};Stouffer Z={Z:+.2f},p={p_st:.4f}**")
    return dict(k=k, n=n, sign_p=sgn, Z=Z, p=p_st)

FCS, RES, ALE = "frontiercs", "frontiercs_research", "alebench"
BENCHES = [FCS, RES, ALE]

print("# 方向一致性聚合(逐格 + 跨格聚合)")
print("\n口径:逐题配对(先按题对 sample 取均值),Wilcoxon 双侧;")
print("聚合 = 格子方向的符号检验 + 逐格 signed Z 的 Stouffer 合并。")

# ---- 1. RL(prior) - RL(base), score -------------------------------------------
c9 = [cell(b, f"rlv5_{s}_s20", "rlv5_base_s20") for b in BENCHES
      for s in ("ft01mix_a10", "ft03nm_a20", "lo32nm_a10")]
aggregate(c9, "1a. 9B:RL(先验) − RL(base),分数")
c4 = [cell(b, f"rlv5_4b_{s}_s20", "rlv5_4b_base_s20") for b in BENCHES
      for s in ("ft01mix_a10", "lo32nm_a10")]
aggregate(c4, "1b. 4B:RL(先验) − RL(base),分数")
aggregate(c9 + c4, "1c. 两个尺度合并")

# ---- 2. same, P(complete) ------------------------------------------------------
k9 = [cell(b, f"rlv5_{s}_s20", "rlv5_base_s20", "complete") for b in BENCHES
      for s in ("ft01mix_a10", "ft03nm_a20", "lo32nm_a10")]
aggregate(k9, "2a. 9B:RL(先验) − RL(base),完成率")
k4 = [cell(b, f"rlv5_4b_{s}_s20", "rlv5_4b_base_s20", "complete") for b in BENCHES
      for s in ("ft01mix_a10", "lo32nm_a10")]
aggregate(k4, "2b. 4B:同上")
aggregate(k9 + k4, "2c. 两个尺度合并")

# ---- 3. SFT - base -------------------------------------------------------------
s9 = [cell(b, s, "base9b_v2c") for b in BENCHES
      for s in ("ft01mix_a10", "ft03nm_a20", "lo32nm_a10")]
aggregate(s9, "3a. 9B:SFT − base,分数")
s4 = [cell(b, s, "base4b") for b in BENCHES for s in ("4b_ft01mix_a10", "4b_lo32nm_a10")]
aggregate(s4, "3b. 4B:SFT − base,分数")

# ---- 4. ft01mix line only (the arm we actually ship) ---------------------------
f9 = [cell(b, "rlv5_ft01mix_a10_s20", "rlv5_base_s20") for b in BENCHES]
f4 = [cell(b, "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_base_s20") for b in BENCHES]
aggregate(f9 + f4, "4. 只看 ft01mix 这条线:RL(先验) − RL(base),分数,6 个格子")

# ---- 5. NOISE FLOOR: replicate arms through the identical machinery ------------
REP = [("base9b_v2c_y2026", "base9b_v2c"), ("lo32nm_a10_y2026", "lo32nm_a10"),
       ("rlv5_lo32nm_a10_s20_y2026", "rlv5_lo32nm_a10_s20"), ("base4b_y2026", "base4b"),
       ("4b_lo32nm_a10_y2026", "4b_lo32nm_a10"),
       ("rlv5_4b_lo32nm_a10_s20_y2026", "rlv5_4b_lo32nm_a10_s20")]
rep = [cell(b, A, B) for b in BENCHES for A, B in REP]
aggregate(rep, "5. 噪声底:同协议复跑 − 主跑(什么都没改),分数")
repc = [cell(b, A, B, "complete") for b in BENCHES for A, B in REP]
aggregate(repc, "5b. 噪声底:同上,完成率")

# ---- 6. the shipped line only, per scale --------------------------------------
aggregate([cell(b, "rlv5_ft01mix_a10_s20", "rlv5_base_s20") for b in BENCHES],
          "6a. 9B ft01mix 线:RL(先验) − RL(base),分数,3 格")
aggregate([cell(b, "rlv5_ft01mix_a10_s20", "rlv5_base_s20", "complete") for b in BENCHES],
          "6b. 9B ft01mix 线:完成率,3 格")
aggregate([cell(b, "rlv5_4b_ft01mix_a10_s20", "rlv5_4b_base_s20", "complete") for b in BENCHES],
          "6c. 4B ft01mix 线:完成率,3 格")

# ---- 7. SFT - base with the confounded ALE-9B cells dropped --------------------
# 9B ALE is judged on mixed node classes (base/SFT on cpu speedFactor 0.79-0.83, RL on
# ailab ~1.0) and ALE is wall-clock scored, so those cells cannot carry an SFT-vs-base
# comparison. This exclusion is the one already documented in the coverage audit; it is
# a topology fact, not a choice made after seeing the numbers.
aggregate([cell(b, s, "base9b_v2c") for b in (FCS, RES)
           for s in ("ft01mix_a10", "ft03nm_a20", "lo32nm_a10")],
          "7a. 9B SFT − base,分数,剔除判题拓扑混杂的 ALE-9B(6 格)")
aggregate([cell(b, "ft01mix_a10", "base9b_v2c") for b in (FCS, RES)],
          "7b. 同上,只看 ft01mix(2 格)")

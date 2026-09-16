#!/usr/bin/env python3
"""Directional aggregation for the INNOVATION-side metrics (not score).

Same machinery as dir_agg.py: per-cell paired-by-problem contrast, then a sign test and
Stouffer combination across cells. Applied to the metrics that are supposed to speak to
innovation rather than to correctness:

  explore_metrics.csv : n_reason / n_abandon / explore_ratio / n_alt  (search breadth
                        BEFORE the answer is fixed -- read from the reasoning, not the code)
  recomb_metrics.csv  : n_tech / n_pair / new_pair_rate / med_z / p10_z  (Uzzi-style
                        recombination of technique families, scored against the official
                        solution pool as the "prior literature")
  metrics.csv         : sim_self (self-similarity across the 5 draws; LOWER = more varied)
                        jac_pool (similarity to the frontier-model pool; LOWER = less derivative)

The replicate arms go through the identical machinery as the noise floor.
"""
import csv, collections, math, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest, norm

def load(path):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(path)):
        try: si = int(r["sample_idx"])
        except Exception: continue
        d[r["bench"]][r["arm"]][(r["problem"], si)] = r
    return d

def cell(DATA, bench, A, B, field):
    da, db = DATA[bench].get(A), DATA[bench].get(B)
    if not da or not db: return None
    byp = collections.defaultdict(lambda: ([], []))
    for k in set(da) & set(db):
        for src, idx in ((da, 0), (db, 1)):
            v = src[k].get(field)
            if v in (None, ""): continue
            if isinstance(v, str) and v.strip() in ("True", "False"):
                v = 1.0 if v.strip() == "True" else 0.0
            try: byp[k[0]][idx].append(float(v))
            except ValueError: return None
    xa, xb = [], []
    for p, (la, lb) in byp.items():
        if la and lb: xa.append(np.mean(la)); xb.append(np.mean(lb))
    if len(xa) < 8: return None
    d = np.array(xa) - np.array(xb); nz = d[d != 0]
    if len(nz) < 5: return None
    try: _, p_two = wilcoxon(nz)
    except Exception: return None
    if not np.isfinite(p_two): return None
    # A Wilcoxon p of exactly 0 (all ranks one way) would send norm.ppf to -inf; the
    # floor caps the per-cell Z at ~7.1 so one saturated cell cannot carry a Stouffer.
    z = (1.0 if np.median(nz) > 0 else -1.0) * abs(norm.ppf(max(p_two, 1e-12) / 2))
    if not np.isfinite(z): return None
    return dict(bench=bench, n=len(d), mean=float(d.mean()),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), p=float(p_two), z=float(z))

def agg(cells, label, better="higher"):
    cells = [c for c in cells if c]
    if not cells:
        print(f"\n**{label}**: 格子不足"); return
    k, n = sum(1 for c in cells if c["mean"] > 0), len(cells)
    sgn = binomtest(k, n, 0.5).pvalue
    Z = sum(c["z"] for c in cells) / math.sqrt(n)
    p = 2 * (1 - norm.cdf(abs(Z)))
    fav = (k if better == "higher" else n - k)
    print(f"| {label} | {n} | {k}/{n} 正 | {sgn:.4f} | {Z:+.2f} | {p:.4f} | "
          f"{'有利' if (Z>0)==(better=='higher') and p<0.05 else ('方向有利' if (Z>0)==(better=='higher') else '方向不利')} |")

FCS, RES, ALE = "frontiercs", "frontiercs_research", "alebench"
BENCHES = [FCS, RES, ALE]
P9 = ["ft01mix_a10", "ft03nm_a20", "lo32nm_a10"]
P4 = ["4b_ft01mix_a10", "4b_lo32nm_a10"]
REP = [("base9b_v2c_y2026", "base9b_v2c"), ("lo32nm_a10_y2026", "lo32nm_a10"),
       ("rlv5_lo32nm_a10_s20_y2026", "rlv5_lo32nm_a10_s20"), ("base4b_y2026", "base4b"),
       ("4b_lo32nm_a10_y2026", "4b_lo32nm_a10"),
       ("rlv5_4b_lo32nm_a10_s20_y2026", "rlv5_4b_lo32nm_a10_s20")]

def block(title, path, fields):
    D = load(path)
    print(f"\n## {title}\n")
    print("| 指标 / 对比 | 格子数 | 方向 | 符号检验 p | Stouffer Z | p | 判读 |")
    print("|---|---|---|---|---|---|---|")
    for f, better in fields:
        agg([cell(D, b, f"rlv5_{s}_s20", "rlv5_base_s20", f) for b in BENCHES for s in P9],
            f"`{f}` 9B RL(先验)−RL(base)", better)
        agg([cell(D, b, "rlv5_"+s.replace("4b_","4b_")+"_s20" if False else f"rlv5_{s}_s20", "rlv5_4b_base_s20", f)
             for b in BENCHES for s in P4], f"`{f}` 4B RL(先验)−RL(base)", better)
        agg([cell(D, b, s, "base9b_v2c", f) for b in BENCHES for s in P9],
            f"`{f}` 9B SFT−base", better)
        agg([cell(D, b, s, "base4b", f) for b in BENCHES for s in P4],
            f"`{f}` 4B SFT−base", better)
        agg([cell(D, b, A, B, f) for b in BENCHES for A, B in REP],
            f"`{f}` **噪声底**(同协议复跑)", better)
    return D

print("# 创新性指标的方向一致性聚合")
print("\n判读列:`better` 方向由指标定义决定(探索量越高越有利;自相似度 / 与解池相似度越低越有利)。")
block("A. 推理过程中的探索(explore_metrics.csv)", "explore_metrics.csv",
      [("n_reason", "higher"), ("n_abandon", "higher"), ("explore_ratio", "higher"),
       ("n_alt", "higher"), ("n_reason_10k", "higher")])
block("B. 技术重组(recomb_metrics.csv)", "recomb_metrics.csv",
      [("n_tech", "higher"), ("n_pair", "higher"), ("new_pair_rate", "higher"),
       ("med_z", "lower"), ("p10_z", "lower")])
block("C. 多样性与派生度(metrics.csv)", "metrics.csv",
      [("sim_self", "lower"), ("jac_pool", "lower")])

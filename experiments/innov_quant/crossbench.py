# -*- coding: utf-8 -*-
"""跨 bench 合并检验:带创新先验的 RL 臂(rl_sft) 是否稳定强于 从 base 直接 RL 的臂(rl_base)?
单个 bench 上多半不显著;这里做逐题配对 + 跨 bench 合并(Stouffer),并数方向一致性(符号检验)。
同时给出 P(complete) 这一通道的同样检验。"""
import os, math, json
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(HERE, "metrics.csv"), low_memory=False)
m = m[m.in_common.astype(str).isin(["True","true","1"])]
m["complete_b"] = m.complete.astype(str).isin(["True","true","1"])

RLB = {"9B": "rlv5_base_s20", "4B": "rlv5_4b_base_s20"}
RLS = {"9B": ["rlv5_ft01mix_a10_s20","rlv5_ft03nm_a20_s20","rlv5_lo32nm_a10_s20"],
       "4B": ["rlv5_4b_ft01mix_a10_s20","rlv5_4b_lo32nm_a10_s20"]}
SFT = {"9B": ["ft01mix_a10","ft03nm_a20","lo32nm_a10"], "4B": ["4b_ft01mix_a10","4b_lo32nm_a10"]}
BASE = {"9B": "base9b_v2c", "4B": "base4b"}
# ALE 9B:rlv5_ft03nm 判在 gpu-ee(speedFactor 0.56),与 ailab 的 rlv5_base 不可比
ALE_BAD = {("alebench","9B","rlv5_ft03nm_a20_s20")}

def paired(sub, a, c, col):
    A = sub[sub.arm == a].groupby("problem")[col].mean()
    C = sub[sub.arm == c].groupby("problem")[col].mean()
    idx = A.index.intersection(C.index)
    if len(idx) < 5: return None
    d = (A[idx] - C[idx]).values
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    try: w = stats.wilcoxon(d, zero_method="wilcox").pvalue
    except Exception: w = float("nan")
    sg = stats.binomtest(pos, pos+neg, 0.5).pvalue if pos+neg else float("nan")
    return dict(n=len(idx), mean=float(d.mean()), pos=pos, neg=neg, wilcox=float(w), sign=float(sg))

def stouffer(ps, signs):
    """单边 z 合并:每个 bench 把双边 p 转成带方向的 z。"""
    zs = []
    for p, s in zip(ps, signs):
        if not np.isfinite(p): continue
        p = min(max(p, 1e-12), 1-1e-12)
        z = stats.norm.isf(p/2) * (1 if s >= 0 else -1)
        zs.append(z)
    if not zs: return float("nan"), float("nan")
    Z = sum(zs)/math.sqrt(len(zs))
    return Z, float(stats.norm.sf(Z))   # 单边:rl_sft > rl_base

BENCHES = ["frontiercs","frontiercs_research","alebench"]
out = []

# ---- MLS:每臂每题一次,直接当第 4 个 bench ----
mls = None
f = os.path.join(os.path.dirname(HERE), "mlsq", "flat.csv")
if os.path.exists(f):
    mm = pd.read_csv(f, low_memory=False)
    mm = mm[mm.era.astype(str).str.contains("old|p1", na=False)]
    # 每个 (arm, task) 取一行:优先 p1
    mm["pri"] = mm.era.astype(str).str.contains("p1").astype(int)
    mm = mm.sort_values("pri").groupby(["arm","task"], as_index=False).last()
    mls = mm

for col, label in [("score","mean@draw"), ("complete_b","P(complete)")]:
    out.append("\n# %s:rl_sft vs rl_base(逐题配对)" % label)
    for fam in ["9B","4B"]:
        out.append("\n## %s" % fam)
        out.append("\n| arm | bench | n | mean Δ | +/− | Wilcoxon p | sign p |")
        out.append("|---|---|---|---|---|---|---|")
        agg = {}
        for a in RLS[fam]:
            ps, sg, rows = [], [], []
            for b in BENCHES:
                if (b, fam, a) in ALE_BAD:
                    out.append("| %s | %s | – | – | – | ⚠判题拓扑不同,剔除 | – |" % (a, b)); continue
                sub = m[(m.bench == b) & (m.fam == fam)]
                r = paired(sub, a, RLB[fam], col)
                if not r: continue
                out.append("| %s | %s | %d | %+.3f | %d/%d | %.3f | %.3f |" %
                           (a, b, r["n"], r["mean"], r["pos"], r["neg"], r["wilcox"], r["sign"]))
                ps.append(r["wilcox"]); sg.append(r["mean"]); rows.append((b, r))
            if col == "score" and mls is not None and fam == "9B":
                am = {"rlv5_ft01mix_a10_s20":"rlv5_ft01mix","rlv5_ft03nm_a20_s20":"rlv5_ft03nm","rlv5_lo32nm_a10_s20":"rlv5_lo32nm"}.get(a)
                cm = "rlv5_base"
                if am and am in set(mls.arm) and cm in set(mls.arm):
                    A = mls[mls.arm == am].set_index("task")["score"]
                    C = mls[mls.arm == cm].set_index("task")["score"]
                    idx = A.index.intersection(C.index)
                    d = (A[idx] - C[idx]).astype(float).values
                    pos, neg = int((d>0).sum()), int((d<0).sum())
                    try: w = stats.wilcoxon(d, zero_method="wilcox").pvalue
                    except Exception: w = float("nan")
                    sgp = stats.binomtest(pos, pos+neg, 0.5).pvalue if pos+neg else float("nan")
                    out.append("| %s | mlsbench | %d | %+.4f | %d/%d | %.3f | %.3f |" % (a, len(idx), d.mean(), pos, neg, w, sgp))
                    ps.append(w); sg.append(d.mean())
            Z, P = stouffer(ps, sg)
            agg[a] = (Z, P, len(ps), sum(1 for x in sg if x > 0), len(sg))
        out.append("\n合并(Stouffer,单边 rl_sft>rl_base):")
        for a, (Z, P, k, npos, ntot) in agg.items():
            out.append("- %s:Z=%.2f,p=%.4f(%d 个 bench;方向为正 %d/%d)" % (a, Z, P, k, npos, ntot))
        allpos = sum(v[3] for v in agg.values()); alltot = sum(v[4] for v in agg.values())
        if alltot:
            out.append("- **全部 (臂 × bench) 方向一致性:%d/%d 为正,符号检验 p=%.4f**" %
                       (allpos, alltot, stats.binomtest(allpos, alltot, 0.5).pvalue))

# ---- 对照组:SFT vs base(RL 之前),同样口径 ----
out.append("\n\n# 对照:SFT vs base(RL 之前),同样口径")
for fam in ["9B","4B"]:
    out.append("\n## %s" % fam)
    out.append("\n| arm | bench | n | mean Δ | +/− | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|")
    ps, sg = [], []
    for a in SFT[fam]:
        for b in BENCHES:
            sub = m[(m.bench == b) & (m.fam == fam)]
            r = paired(sub, a, BASE[fam], "score")
            if not r: continue
            out.append("| %s | %s | %d | %+.3f | %d/%d | %.3f |" % (a, b, r["n"], r["mean"], r["pos"], r["neg"], r["wilcox"]))
            ps.append(r["wilcox"]); sg.append(r["mean"])
    if sg:
        Z, P = stouffer(ps, sg)
        out.append("\n- 合并 Z=%.2f,p=%.4f;方向为正 %d/%d,符号检验 p=%.4f" %
                   (Z, P, sum(1 for x in sg if x>0), len(sg), stats.binomtest(sum(1 for x in sg if x>0), len(sg), 0.5).pvalue))

print("\n".join(out))

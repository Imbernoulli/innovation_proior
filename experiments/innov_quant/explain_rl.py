# -*- coding: utf-8 -*-
"""为什么 RL 之后分数更高?把 mean@5 的增益拆成三个通道:
   完成率 p_c、完成样本里非零率 p_pos、非零样本的条件分 e_pos。
   再看:非零样本的分数分布是整体上移,还是只是 0 变非零。
"""
import os, csv, math, statistics as st
import pandas as pd, numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(HERE, "metrics.csv"))
o = pd.read_csv(os.path.join(HERE, "oracle_metrics.csv"))
m = m[m.in_common == True] if m.in_common.dtype == bool else m[m.in_common.astype(str).isin(["True","true","1"])]

ARMS = {}
for _, r in m[["arm","stage","fam"]].drop_duplicates().iterrows():
    ARMS[r["arm"]] = (r["fam"], r["stage"])

PAIRS = [
 ("ft01mix_a10","base9b_v2c"),("ft03nm_a20","base9b_v2c"),("lo32nm_a10","base9b_v2c"),
 ("rlv5_base_s20","base9b_v2c"),
 ("rlv5_ft01mix_a10_s20","ft01mix_a10"),("rlv5_ft03nm_a20_s20","ft03nm_a20"),("rlv5_lo32nm_a10_s20","lo32nm_a10"),
 ("rlv5_ft01mix_a10_s20","rlv5_base_s20"),("rlv5_ft03nm_a20_s20","rlv5_base_s20"),("rlv5_lo32nm_a10_s20","rlv5_base_s20"),
 ("4b_ft01mix_a10","base4b"),("4b_lo32nm_a10","base4b"),("rlv5_4b_base_s20","base4b"),
 ("rlv5_4b_ft01mix_a10_s20","4b_ft01mix_a10"),("rlv5_4b_lo32nm_a10_s20","4b_lo32nm_a10"),
 ("base9b_v2c_y2026","base9b_v2c"),("lo32nm_a10_y2026","lo32nm_a10"),
 ("rlv5_lo32nm_a10_s20_y2026","rlv5_lo32nm_a10_s20"),
]

def factors(g):
    n = len(g)
    if n == 0: return None
    c = g[g.complete == True] if g.complete.dtype == bool else g[g.complete.astype(str).isin(["True","true","1"])]
    p_c = len(c)/n
    tr = g.drop(c.index)
    e_tr = tr.score.mean() if len(tr) else 0.0
    if len(c) == 0: return dict(n=n, p_c=0.0, p_pos=float("nan"), e_pos=float("nan"), e_tr=e_tr, mean=g.score.mean())
    pos = c[c.score > 0]
    p_pos = len(pos)/len(c)
    e_pos = pos.score.mean() if len(pos) else 0.0
    return dict(n=n, p_c=p_c, p_pos=p_pos, e_pos=e_pos, e_tr=e_tr, mean=g.score.mean())

out = []
for bench in ["frontiercs","frontiercs_research","alebench"]:
    for fam in ["9B","4B"]:
        sub = m[(m.bench == bench) & (m.fam == fam)]
        if len(sub) == 0: continue
        out.append("\n## %s / %s" % (bench, fam))
        out.append("\n### A. 三通道分解(所有 common 抽样合并)")
        out.append("\n| arm | stage | n draws | P(complete) | P(score>0 \\| complete) | E[score \\| complete,>0] | E[score \\| truncated] | mean@draw |")
        out.append("|---|---|---|---|---|---|---|---|")
        for a in sorted(sub.arm.unique(), key=lambda x: (ARMS[x][1], x)):
            f = factors(sub[sub.arm == a])
            out.append("| %s | %s | %d | %.3f | %.3f | %.2f | %.2f | %.2f |" %
                       (a, ARMS[a][1], f["n"], f["p_c"], f["p_pos"], f["e_pos"], f["e_tr"], f["mean"]))

        out.append("\n### B. 配对增益分解 Δmean ≈ Δp_c·B̄·C̄ + p̄_c·Δp_pos·C̄ + p̄_c·p̄_pos·Δe_pos + 截断项")
        out.append("\n(逐题算三因子后按题平均;残差 = 交叉项。)")
        out.append("\n| pair | n probs | Δmean | 完成率通道 | 非零率通道 | 条件分通道 | 截断项 | 残差 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for a, c in PAIRS:
            if a not in set(sub.arm) or c not in set(sub.arm): continue
            probs = sorted(set(sub[sub.arm == a].problem) & set(sub[sub.arm == c].problem))
            rows = []
            for p in probs:
                fa = factors(sub[(sub.arm == a) & (sub.problem == p)])
                fc = factors(sub[(sub.arm == c) & (sub.problem == p)])
                if fa is None or fc is None: continue
                # 用 0 填补无 complete 的情形
                pa, pc_ = fa["p_c"], fc["p_c"]
                qa = 0.0 if math.isnan(fa["p_pos"]) else fa["p_pos"]
                qc = 0.0 if math.isnan(fc["p_pos"]) else fc["p_pos"]
                ea = 0.0 if math.isnan(fa["e_pos"]) else fa["e_pos"]
                ec = 0.0 if math.isnan(fc["e_pos"]) else fc["e_pos"]
                pb, qb, eb = (pa+pc_)/2, (qa+qc)/2, (ea+ec)/2
                t1 = (pa-pc_)*qb*eb
                t2 = pb*(qa-qc)*eb
                t3 = pb*qb*(ea-ec)
                ttr = (1-pa)*fa["e_tr"] - (1-pc_)*fc["e_tr"]
                dm = fa["mean"] - fc["mean"]
                rows.append((dm, t1, t2, t3, ttr, dm-(t1+t2+t3+ttr)))
            if not rows: continue
            A = np.array(rows)
            out.append("| %s − %s | %d | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f |" %
                       (a, c, len(rows), *A.mean(axis=0)))

        out.append("\n### C. 非零样本的分数分布(complete & score>0):是整体上移还是只是多了几个非零?")
        out.append("\n| arm | n pos | p25 | median | p75 | p90 | max |")
        out.append("|---|---|---|---|---|---|---|")
        for a in sorted(sub.arm.unique(), key=lambda x: (ARMS[x][1], x)):
            g = sub[(sub.arm == a)]
            cc = g[g.complete == True] if g.complete.dtype == bool else g[g.complete.astype(str).isin(["True","true","1"])]
            pos = cc[cc.score > 0].score.values
            if len(pos) == 0: continue
            out.append("| %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f |" %
                       (a, len(pos), np.percentile(pos,25), np.median(pos), np.percentile(pos,75),
                        np.percentile(pos,90), pos.max()))

print("\n".join(out))

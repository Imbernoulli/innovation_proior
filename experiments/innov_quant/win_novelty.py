# -*- coding: utf-8 -*-
"""“赢的时候是不是靠不一样的方法赢的?”
只看在该题成功的抽样(score >= 该题所有臂样本的 75 分位,且 >0),比较各臂这些成功解与 frontier 解池的相似度。
若创新先验有用,它的成功解应当**更不像**解池(jac_frontier 更低)而分数不低。"""
import os
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
o = pd.read_csv(os.path.join(HERE, "oracle_metrics.csv"), low_memory=False)
o = o[o.in_common.astype(str).isin(["True","true","1"])]
o["problem"] = o["problem"].astype(str)
o = o[o.jac_frontier.notna()]
PAIRS = [("ft01mix_a10","base9b_v2c"),("ft03nm_a20","base9b_v2c"),("lo32nm_a10","base9b_v2c"),
 ("rlv5_base_s20","base9b_v2c"),("rlv5_ft01mix_a10_s20","ft01mix_a10"),
 ("rlv5_ft03nm_a20_s20","ft03nm_a20"),("rlv5_lo32nm_a10_s20","lo32nm_a10"),
 ("rlv5_ft01mix_a10_s20","rlv5_base_s20"),("rlv5_ft03nm_a20_s20","rlv5_base_s20"),
 ("rlv5_lo32nm_a10_s20","rlv5_base_s20"),
 ("4b_ft01mix_a10","base4b"),("4b_lo32nm_a10","base4b"),
 ("rlv5_4b_ft01mix_a10_s20","4b_ft01mix_a10"),("rlv5_4b_lo32nm_a10_s20","4b_lo32nm_a10"),
 ("base9b_v2c_y2026","base9b_v2c"),("lo32nm_a10_y2026","lo32nm_a10"),
 ("rlv5_lo32nm_a10_s20_y2026","rlv5_lo32nm_a10_s20")]
out = []
for bench in ["frontiercs","frontiercs_research"]:
    for fam in ["9B","4B"]:
        sub = o[(o.bench == bench) & (o.fam == fam)].copy()
        if len(sub) < 200: continue
        thr = sub.groupby("problem").score.quantile(0.75)
        sub["thr"] = sub.problem.map(thr)
        win = sub[(sub.score > 0) & (sub.score >= sub.thr)]
        out.append("\n## %s / %s(成功解 = 该题分数 >0 且 >= 全臂 75 分位;共 %d 个)" % (bench, fam, len(win)))
        out.append("\n| arm | n 成功解 | 覆盖题数 | mean jac_frontier(成功解) | mean jac_frontier(全部) |")
        out.append("|---|---|---|---|---|")
        for a in sorted(sub.arm.unique()):
            g = win[win.arm == a]; ga = sub[sub.arm == a]
            if len(g) == 0: continue
            out.append("| %s | %d | %d | %.3f | %.3f |" % (a, len(g), g.problem.nunique(), g.jac_frontier.mean(), ga.jac_frontier.mean()))
        out.append("\n逐题配对(只用两臂都有成功解的题):")
        out.append("\n| pair | n probs | Δ jac_frontier(成功解) (+/−) | Wilcoxon p |")
        out.append("|---|---|---|---|")
        for A, C in PAIRS:
            if A not in set(win.arm) or C not in set(win.arm): continue
            x = win[win.arm == A].groupby("problem").jac_frontier.mean()
            y = win[win.arm == C].groupby("problem").jac_frontier.mean()
            idx = x.index.intersection(y.index); v = (x[idx]-y[idx]).dropna()
            if len(v) < 5: continue
            try: p = stats.wilcoxon(v.values, zero_method="wilcox").pvalue
            except Exception: p = float("nan")
            out.append("| %s − %s | %d | %+.4f (%d/%d) | %.3f |" % (A, C, len(v), v.mean(), int((v>0).sum()), int((v<0).sum()), p))
print("\n".join(out))

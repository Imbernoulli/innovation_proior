# -*- coding: utf-8 -*-
"""长度控制:RL 之后代码更短,技术家族数下降会不会只是“字少了扫不到词”?
两个控制:(1) 密度 n_tech/100LOC;(2) 把 n_tech 对 log(1+LOC) 做全局 OLS 后取残差,再逐题配对。"""
import os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "recomb_metrics.csv"), low_memory=False)
m = pd.read_csv(os.path.join(HERE, "metrics.csv"), low_memory=False)[
    ["bench","arm","problem","sample_idx","loc","n_tokens"]]
for x in (d, m): x["problem"] = x["problem"].astype(str)
d = d.merge(m, on=["bench","arm","problem","sample_idx"], how="left")
d = d[d["loc"].notna() & (d["loc"] > 0)]
d["dens"] = 100.0 * d.n_tech / d["loc"]
d["lloc"] = np.log1p(d["loc"])

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
for bench in ["frontiercs","frontiercs_research","alebench"]:
    for fam in ["9B","4B"]:
        sub = d[(d.bench == bench) & (d.fam == fam)].copy()
        if len(sub) < 200: continue
        # 全局 OLS: n_tech ~ a + b*log(1+LOC)
        b, a = np.polyfit(sub.lloc.values, sub.n_tech.values, 1)
        sub["resid"] = sub.n_tech - (a + b*sub.lloc)
        out.append("\n## %s / %s  (n_tech ~ %.3f + %.3f·log(1+LOC), n=%d)" % (bench, fam, a, b, len(sub)))
        out.append("\n| arm | med LOC | mean n_tech | mean n_tech/100LOC | mean 残差 |")
        out.append("|---|---|---|---|---|")
        for arm in sorted(sub.arm.unique()):
            g = sub[sub.arm == arm]
            out.append("| %s | %.0f | %.2f | %.2f | %+.3f |" % (arm, g["loc"].median(), g.n_tech.mean(), g.dens.mean(), g.resid.mean()))
        out.append("\n| pair | n probs | Δ n_tech p | Δ 密度 p | Δ 残差 (+/−) p |")
        out.append("|---|---|---|---|---|")
        for A, C in PAIRS:
            if A not in set(sub.arm) or C not in set(sub.arm): continue
            cells = []; n = 0
            for col in ["n_tech","dens","resid"]:
                x = sub[sub.arm == A].groupby("problem")[col].mean()
                y = sub[sub.arm == C].groupby("problem")[col].mean()
                idx = x.index.intersection(y.index); v = (x[idx]-y[idx]).dropna(); n = max(n, len(v))
                if len(v) < 5: cells.append("–"); continue
                try: p = stats.wilcoxon(v.values, zero_method="wilcox").pvalue
                except Exception: p = float("nan")
                cells.append("%+.3f (%d/%d) %.3f" % (v.mean(), int((v>0).sum()), int((v<0).sum()), p))
            out.append("| %s − %s | %d | %s |" % (A, C, n, " | ".join(cells)))
print("\n".join(out))

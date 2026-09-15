# -*- coding: utf-8 -*-
"""推理过程层面的探索指标:每臂表 + 逐题配对 + 题内相关。"""
import os
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "explore_metrics.csv"), low_memory=False)
d["problem"] = d["problem"].astype(str)
ST = {}
for _, r in d[["arm","stage"]].drop_duplicates().iterrows(): ST[r["arm"]] = r["stage"]
ORDER = {"base":0,"sft":1,"rl_base":2,"rl_sft":3,"rep":4}
PAIRS = [("ft01mix_a10","base9b_v2c"),("ft03nm_a20","base9b_v2c"),("lo32nm_a10","base9b_v2c"),
 ("rlv5_base_s20","base9b_v2c"),("rlv5_ft01mix_a10_s20","ft01mix_a10"),
 ("rlv5_ft03nm_a20_s20","ft03nm_a20"),("rlv5_lo32nm_a10_s20","lo32nm_a10"),
 ("rlv5_ft01mix_a10_s20","rlv5_base_s20"),("rlv5_ft03nm_a20_s20","rlv5_base_s20"),
 ("rlv5_lo32nm_a10_s20","rlv5_base_s20"),
 ("4b_ft01mix_a10","base4b"),("4b_lo32nm_a10","base4b"),
 ("rlv5_4b_ft01mix_a10_s20","4b_ft01mix_a10"),("rlv5_4b_lo32nm_a10_s20","4b_lo32nm_a10"),
 ("rlv5_4b_ft01mix_a10_s20","rlv5_4b_base_s20"),("rlv5_4b_lo32nm_a10_s20","rlv5_4b_base_s20"),
 ("base9b_v2c_y2026","base9b_v2c"),("lo32nm_a10_y2026","lo32nm_a10"),
 ("rlv5_lo32nm_a10_s20_y2026","rlv5_lo32nm_a10_s20"),
 ("base4b_y2026","base4b"),("4b_lo32nm_a10_y2026","4b_lo32nm_a10"),
 ("rlv5_4b_lo32nm_a10_s20_y2026","rlv5_4b_lo32nm_a10_s20")]
COLS = ["n_reason","n_abandon","explore_ratio","n_alt","n_reason_10k","n_abandon_10k","n_alt_10k"]
out = []
for bench in ["frontiercs","frontiercs_research","alebench"]:
    for fam in ["9B","4B"]:
        sub = d[(d.bench == bench) & (d.fam == fam)]
        if len(sub) < 100: continue
        out.append("\n## %s / %s" % (bench, fam))
        out.append("\n### A. 每臂(仅 complete 且有代码的样本)")
        out.append("\n| arm | stage | n | med 推理字符 | n_reason | n_code | n_abandon | explore_ratio | n_alt | n_reason/10k | n_abandon/10k | n_alt/10k |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in sorted(sub.arm.unique(), key=lambda x: (ORDER.get(ST[x],9), x)):
            g = sub[sub.arm == a]
            out.append("| %s | %s | %d | %.0f | %.2f | %.2f | %.2f | %.2f | %.1f | %.3f | %.3f | %.3f |" % (
                a, ST[a], len(g), g.rlen.median(), g.n_reason.mean(), g.n_code.mean(), g.n_abandon.mean(),
                g.explore_ratio.mean(), g.n_alt.mean(), g.n_reason_10k.mean(), g.n_abandon_10k.mean(), g.n_alt_10k.mean()))
        out.append("\n### B. 逐题配对")
        out.append("\n| pair | n | Δ n_reason p | Δ n_abandon p | Δ explore_ratio p | Δ n_alt p | Δ n_reason/10k p | Δ n_alt/10k p |")
        out.append("|---|---|---|---|---|---|---|---|")
        for A, C in PAIRS:
            if A not in set(sub.arm) or C not in set(sub.arm): continue
            cells = []; n = 0
            for col in ["n_reason","n_abandon","explore_ratio","n_alt","n_reason_10k","n_alt_10k"]:
                x = sub[sub.arm == A].groupby("problem")[col].mean()
                y = sub[sub.arm == C].groupby("problem")[col].mean()
                idx = x.index.intersection(y.index); v = (x[idx]-y[idx]).dropna(); n = max(n, len(v))
                if len(v) < 5: cells.append("–"); continue
                try: p = stats.wilcoxon(v.values, zero_method="wilcox").pvalue
                except Exception: p = float("nan")
                cells.append("%+.3f (%d/%d) %.3f" % (v.mean(), int((v>0).sum()), int((v<0).sum()), p))
            out.append("| %s − %s | %d | %s |" % (A, C, n, " | ".join(cells)))
        out.append("\n### C. 题内关联:探索指标 ↔ 分数(main 臂)")
        mn = sub[sub.arm.map(lambda x: ST[x]) != "rep"]
        out.append("\n| property | n | rho | p |")
        out.append("|---|---|---|---|")
        for col in COLS:
            g = mn.dropna(subset=[col])
            if len(g) < 100: continue
            r = g.groupby("problem")[[col,"score"]].rank(pct=True)
            rho, p = stats.spearmanr(r[col], r["score"])
            out.append("| %s | %d | %+.3f | %.4f |" % (col, len(g), rho, p))
print("\n".join(out))

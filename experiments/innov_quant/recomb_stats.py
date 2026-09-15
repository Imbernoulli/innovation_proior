# -*- coding: utf-8 -*-
"""重组指标的每臂表 + 逐题配对检验 + 题内相关。
指标含义(都只算 complete 且有代码的 common 样本):
  n_tech        一份解里用到的不同技术家族数            低 = 拼的零件少
  P(n_tech>=2)  真正发生组合的比例
  P(n_tech>=5)  重度组合的比例
  new_pair_rate 该解的技术对中在 frontier 解池从未共现的比例   高 = 新组合
  med_z         Uzzi 常规性:该解技术对的 z 中位数        高 = 都是常规搭配
  p10_z         Uzzi 原子性尾部:z 的 10 分位            低 = 伸手够冷门搭配
"""
import os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "recomb_metrics.csv"), low_memory=False)
d["problem"] = d["problem"].astype(str)

ARMS = {}
for _, r in d[["arm","stage","fam"]].drop_duplicates().iterrows(): ARMS[r["arm"]] = r["stage"]
ORDER = {"base":0,"sft":1,"rl_base":2,"rl_sft":3,"rep":4}
PAIRS = [
 ("ft01mix_a10","base9b_v2c"),("ft03nm_a20","base9b_v2c"),("lo32nm_a10","base9b_v2c"),
 ("rlv5_base_s20","base9b_v2c"),
 ("rlv5_ft01mix_a10_s20","ft01mix_a10"),("rlv5_ft03nm_a20_s20","ft03nm_a20"),("rlv5_lo32nm_a10_s20","lo32nm_a10"),
 ("rlv5_ft01mix_a10_s20","rlv5_base_s20"),("rlv5_ft03nm_a20_s20","rlv5_base_s20"),("rlv5_lo32nm_a10_s20","rlv5_base_s20"),
 ("4b_ft01mix_a10","base4b"),("4b_lo32nm_a10","base4b"),("rlv5_4b_base_s20","base4b"),
 ("rlv5_4b_ft01mix_a10_s20","4b_ft01mix_a10"),("rlv5_4b_lo32nm_a10_s20","4b_lo32nm_a10"),
 ("rlv5_4b_ft01mix_a10_s20","rlv5_4b_base_s20"),("rlv5_4b_lo32nm_a10_s20","rlv5_4b_base_s20"),
 ("base9b_v2c_y2026","base9b_v2c"),("lo32nm_a10_y2026","lo32nm_a10"),
 ("rlv5_lo32nm_a10_s20_y2026","rlv5_lo32nm_a10_s20"),
 ("base4b_y2026","base4b"),("4b_lo32nm_a10_y2026","4b_lo32nm_a10"),
 ("rlv5_4b_lo32nm_a10_s20_y2026","rlv5_4b_lo32nm_a10_s20"),
]
COLS = [("n_tech","n_tech"),("new_pair_rate","new_pair_rate"),("med_z","med_z"),("p10_z","p10_z")]
out = []
for bench in ["frontiercs","frontiercs_research","alebench"]:
    out.append("\n# %s" % bench)
    for fam in ["9B","4B"]:
        sub = d[(d.bench == bench) & (d.fam == fam)]
        if len(sub) == 0: continue
        out.append("\n## %s" % fam)
        out.append("\n### A. 每臂(complete 且有代码的样本)")
        out.append("\n| arm | stage | n | mean n_tech | med n_tech | P(>=2) | P(>=5) | mean new_pair_rate | mean med_z | mean p10_z |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for a in sorted(sub.arm.unique(), key=lambda x: (ORDER.get(ARMS[x],9), x)):
            g = sub[sub.arm == a]
            out.append("| %s | %s | %d | %.2f | %.0f | %.0f%% | %.0f%% | %.3f | %+.2f | %+.2f |" % (
                a, ARMS[a], len(g), g.n_tech.mean(), g.n_tech.median(),
                100*(g.n_tech >= 2).mean(), 100*(g.n_tech >= 5).mean(),
                g.new_pair_rate.mean(skipna=True), g.med_z.mean(skipna=True), g.p10_z.mean(skipna=True)))

        out.append("\n### B. 逐题配对(每题先按臂取样本均值)")
        out.append("\n| pair | n probs | Δ n_tech (+/−) p | Δ new_pair_rate (+/−) p | Δ med_z (+/−) p | Δ p10_z (+/−) p |")
        out.append("|---|---|---|---|---|---|")
        for a, c in PAIRS:
            if a not in set(sub.arm) or c not in set(sub.arm): continue
            cells, n = [], 0
            for col, _ in COLS:
                A = sub[sub.arm == a].groupby("problem")[col].mean()
                C = sub[sub.arm == c].groupby("problem")[col].mean()
                idx = A.index.intersection(C.index)
                v = (A[idx] - C[idx]).dropna()
                n = max(n, len(v))
                if len(v) < 5: cells.append("–"); continue
                pos, neg = int((v > 0).sum()), int((v < 0).sum())
                try: p = stats.wilcoxon(v.values, zero_method="wilcox").pvalue
                except Exception: p = float("nan")
                cells.append("%+.3f (%d/%d) %.3f" % (v.mean(), pos, neg, p))
            out.append("| %s − %s | %d | %s |" % (a, c, n, " | ".join(cells)))

        out.append("\n### C. 题内关联(main 臂样本在题内排秩后合并):重组指标 ↔ 分数")
        mn = sub[sub.arm.map(lambda x: ARMS[x]) != "rep"]
        out.append("\n| property | n | Spearman rho vs score | p |")
        out.append("|---|---|---|---|")
        for col, _ in COLS:
            g = mn.dropna(subset=[col])
            if len(g) < 50: continue
            r = g.groupby("problem")[[col,"score"]].rank(pct=True)
            rho, p = stats.spearmanr(r[col], r["score"])
            out.append("| %s | %d | %+.3f | %.4f |" % (col, len(g), rho, p))
print("\n".join(out))

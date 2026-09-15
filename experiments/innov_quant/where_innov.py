# -*- coding: utf-8 -*-
"""创新如果起作用,应该起在哪儿?四个有针对性的检验:
 1) base-hard 子集:base 臂 best@5 == 0 的题(标准路子根本打不开的题)上,该臂 vs 对照的胜率/增益,
    与 base-easy 子集对照。创新先验若有用,增益应集中在 hard 侧。
 2) 独占解:该题只有这一条臂拿到分(best@5>0)/拿到高分(best@5>=50)。
 3) 尾部:出现一次 score>50 / >80 抽样的题占比(创新体现在尾部而不是均值)。
 4) 超人类:score_unbounded>=100 的题数。
"""
import os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(HERE, "metrics.csv"), low_memory=False)
m = m[m.in_common.astype(str).isin(["True","true","1"])]
m["problem"] = m["problem"].astype(str)

FAMBASE = {"9B": "base9b_v2c", "4B": "base4b"}
PAIRS = {
 "9B": [("ft01mix_a10","base9b_v2c"),("ft03nm_a20","base9b_v2c"),("lo32nm_a10","base9b_v2c"),
        ("rlv5_base_s20","base9b_v2c"),("rlv5_ft01mix_a10_s20","rlv5_base_s20"),
        ("rlv5_ft03nm_a20_s20","rlv5_base_s20"),("rlv5_lo32nm_a10_s20","rlv5_base_s20"),
        ("base9b_v2c_y2026","base9b_v2c"),("rlv5_lo32nm_a10_s20_y2026","rlv5_lo32nm_a10_s20")],
 "4B": [("4b_ft01mix_a10","base4b"),("4b_lo32nm_a10","base4b"),("rlv5_4b_base_s20","base4b"),
        ("rlv5_4b_ft01mix_a10_s20","rlv5_4b_base_s20"),("rlv5_4b_lo32nm_a10_s20","rlv5_4b_base_s20"),
        ("base4b_y2026","base4b")],
}
ALE_BAD = {("alebench","9B","rlv5_ft03nm_a20_s20")}
out = []
for bench in ["frontiercs","frontiercs_research","alebench"]:
    HI = 50 if bench != "alebench" else None   # ALE 分数量纲不同,尾部用分位数
    for fam in ["9B","4B"]:
        sub = m[(m.bench == bench) & (m.fam == fam)]
        if len(sub) == 0: continue
        best = sub.groupby(["arm","problem"]).score.max().unstack(0)
        mean = sub.groupby(["arm","problem"]).score.mean().unstack(0)
        b = FAMBASE[fam]
        # 难度必须用**独立的一次跑**来定义,否则“base=0 的题”是按对照臂自己的抽样选出来的,
        # 任何臂在这个子集上都会 0/N 全胜(复跑臂也一样),那是选择效应不是创新。
        hb = {"9B": "base9b_v2c_y2026", "4B": "base4b_y2026"}[fam]
        if b not in best.columns or hb not in best.columns: continue
        hard = best.index[best[hb].fillna(0) <= 0]
        easy = best.index[best[hb].fillna(0) > 0]
        if bench == "alebench":
            q = best[hb].quantile(0.33)
            hard = best.index[best[hb] <= q]; easy = best.index[best[hb] > q]
        hard = hard.intersection(best.index); easy = easy.intersection(best.index)
        out.append("\n## %s / %s  (难度由独立复跑臂定义:hard %d 题 / easy %d 题)" % (bench, fam, len(hard), len(easy)))
        out.append("\n### 1. 增益是否集中在 base 打不开的题上")
        out.append("\n| pair | hard: mean Δ (+/−) p | easy: mean Δ (+/−) p | hard−easy 差 (Mann-Whitney p) |")
        out.append("|---|---|---|---|")
        for A, C in PAIRS[fam]:
            if (bench, fam, A) in ALE_BAD: continue
            if A not in mean.columns or C not in mean.columns: continue
            cells = []; dd = {}
            for name, idx in [("hard", hard), ("easy", easy)]:
                v = (mean.loc[idx, A] - mean.loc[idx, C]).dropna()
                dd[name] = v
                if len(v) < 5: cells.append("–"); continue
                try: p = stats.wilcoxon(v.values, zero_method="wilcox").pvalue
                except Exception: p = float("nan")
                cells.append("%+.2f (%d/%d) %.3f" % (v.mean(), int((v>0).sum()), int((v<0).sum()), p))
            if len(dd.get("hard",[])) >= 5 and len(dd.get("easy",[])) >= 5:
                u = stats.mannwhitneyu(dd["hard"].values, dd["easy"].values).pvalue
                cells.append("%+.2f (p=%.3f)" % (dd["hard"].mean()-dd["easy"].mean(), u))
            else: cells.append("–")
            out.append("| %s − %s | %s |" % (A, C, " | ".join(cells)))

        out.append("\n### 2/3/4. 独占解、尾部、超人类")
        uni_any = {}; uni_hi = {}
        cols = [c for c in best.columns]
        for p_ in best.index:
            row = best.loc[p_]
            got = [c for c in cols if (row.get(c) or 0) > 0]
            if len(got) == 1: uni_any[got[0]] = uni_any.get(got[0], 0) + 1
            gothi = [c for c in cols if (row.get(c) or 0) >= 50]
            if len(gothi) == 1: uni_hi[gothi[0]] = uni_hi.get(gothi[0], 0) + 1
        out.append("\n| arm | 独占有分题 | 独占高分题(>=50) | P(某题出现 score>50 的抽样) | P(>80) | 超人类题数(unbounded>=100) |")
        out.append("|---|---|---|---|---|---|")
        for a in cols:
            g = sub[sub.arm == a]
            nb = g.problem.nunique()
            t50 = g.groupby("problem").score.max().gt(50).mean()
            t80 = g.groupby("problem").score.max().gt(80).mean()
            sh = "–"
            if "score_unbounded" in g.columns and bench != "alebench":
                sh = str(int(g.groupby("problem").score_unbounded.max().ge(100).sum()))
            out.append("| %s | %d | %d | %.0f%% | %.0f%% | %s |" %
                       (a, uni_any.get(a,0), uni_hi.get(a,0), 100*t50, 100*t80, sh))
print("\n".join(out))

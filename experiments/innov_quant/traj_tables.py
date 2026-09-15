# -*- coding: utf-8 -*-
"""RL 轨迹表:每个血统(base / ft01mix / ft03nm / lo32nm)沿 step 的曲线。"""
import pandas as pd, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "traj_metrics.csv"))
out = []
for bench in ["frontiercs","frontiercs_research","alebench"]:
    for fam in ["9B","4B"]:
        sub = d[(d.bench == bench) & (d.fam == fam)]
        if len(sub) == 0: continue
        out.append("\n## %s / %s" % (bench, fam))
        for col, lab in [("mean_score","mean@draw"),("p_complete","P(complete)"),
                         ("self_sim","自相似度(越低=5 次抽样越不一样)"),
                         ("mean_ntech","技术家族数"),("med_loc","代码 LOC 中位"),("med_ctok","completion tokens 中位")]:
            steps = sorted(sub.step.unique())
            out.append("\n**%s**\n" % lab)
            out.append("| lineage | " + " | ".join("s%d" % s for s in steps) + " |")
            out.append("|---"*(len(steps)+1) + "|")
            for lin in ["base","ft01mix","ft03nm","lo32nm"]:
                g = sub[sub.lineage == lin]
                if len(g) == 0: continue
                cells = []
                for s in steps:
                    r = g[g.step == s]
                    cells.append("–" if len(r) == 0 or pd.isna(r[col].iloc[0]) else ("%.3f" % r[col].iloc[0] if col in ("p_complete","self_sim") else "%.1f" % r[col].iloc[0]))
                out.append("| %s | " % lin + " | ".join(cells) + " |")
open(os.path.join(HERE, "traj_tables.md"), "w").write("\n".join(out))
print("\n".join(out))

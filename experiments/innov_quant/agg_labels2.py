# -*- coding: utf-8 -*-
"""盲标签的**按臂**汇总:每条臂自己的样本被判为 textbook / variant / different / none 的比例,
以及它赢的时候被判为“换了机制”(why=different_method)的比例、被判跨领域的比例。
标注者不知道哪个字母是哪条臂,也没打开过 key.json。"""
import json, glob, os, collections
from scipy import stats
B = "blind"; key = json.load(open(f"{B}/key.json"))
rows = []; bad = 0; n = 0
for f in sorted(glob.glob(f"{B}/labels/*.jsonl")):
    for l in open(f):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        n += 1
        fol = r["folder"].strip("/")
        if fol not in key: continue
        ok = True
        for letter, q in (r.get("quotes") or {}).items():
            p = f"{B}/{fol}/{letter}.txt"
            if not os.path.exists(p) or q not in open(p).read(): ok = False
        if not ok: bad += 1; continue
        k = key[fol]
        bench = fol.split("/", 1)[0]
        kind = k.get("kind", "main"); side = k["winner_side"]
        pair = fol.split("/")[1]; a, c = pair.split("__vs__")
        L2A = {ltr: v["arm"] for ltr, v in k.items() if isinstance(v, dict) and "arm" in v}
        nov = r.get("novel") or {}; cd = r.get("cross_domain") or {}
        why = r.get("why_higher_scores")
        winner = None
        for ltr, arm in L2A.items():
            if (side == "arm" and arm == a) or (side == "ctrl" and arm == c): winner = ltr
        for ltr, arm in L2A.items():
            rows.append(dict(bench=bench, kind=kind, arm=arm, letter=ltr, novel=nov.get(ltr),
                             cross=bool(cd.get(ltr)), won=(ltr == winner), why=why,
                             same_core=bool(r.get("same_core_idea"))))
print(f"labels {n}, usable rows {len(rows)}, quote-failed records dropped {bad}")

from dump2 import ARMS as _A
ST = {a: v[1] for a, v in _A.items()}

out = []
for bench in ["frontiercs","frontiercs_research","alebench"]:
    sub = [r for r in rows if r["bench"] == bench]
    if not sub: continue
    out.append("\n## %s (n=%d 个被标注的样本)" % (bench, len(sub)))
    out.append("\n| arm | stage | n | textbook | variant | **different** | none | 跨领域 | 赢且被判换机制 / 赢 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    arms = sorted({r["arm"] for r in sub}, key=lambda a: (ST.get(a, "z"), a))
    tot = collections.Counter()
    for a in arms:
        g = [r for r in sub if r["arm"] == a]
        c = collections.Counter(r["novel"] for r in g)
        w = [r for r in g if r["won"]]
        wd = sum(1 for r in w if r["why"] == "different_method")
        out.append("| %s | %s | %d | %d | %d | **%d (%.0f%%)** | %d | %d | %d/%d |" % (
            a, ST.get(a,"?"), len(g), c.get("textbook",0), c.get("variant",0), c.get("different",0),
            100*c.get("different",0)/max(1,len(g)), c.get("none",0), sum(1 for r in g if r["cross"]), wd, len(w)))
    # stage 级别的 different 比例 + 卡方
    out.append("\n按 stage 合并 different 比例:")
    for st in ["base","sft","rl_base","rl_sft","rep"]:
        g = [r for r in sub if ST.get(r["arm"]) == st]
        if not g: continue
        out.append("- %s: %d/%d = %.0f%%" % (st, sum(1 for r in g if r["novel"]=="different"), len(g),
                                             100*sum(1 for r in g if r["novel"]=="different")/len(g)))
    for s1, s2 in [("sft","base"),("rl_sft","rl_base"),("rl_sft","sft")]:
        g1 = [r for r in sub if ST.get(r["arm"]) == s1]; g2 = [r for r in sub if ST.get(r["arm"]) == s2]
        if len(g1) < 20 or len(g2) < 20: continue
        a1 = sum(1 for r in g1 if r["novel"]=="different"); a2 = sum(1 for r in g2 if r["novel"]=="different")
        p = stats.fisher_exact([[a1, len(g1)-a1],[a2, len(g2)-a2]])[1]
        out.append("- %s vs %s:%d/%d vs %d/%d,Fisher p=%.3f" % (s1, s2, a1, len(g1), a2, len(g2), p))
open("label_tables.md","w").write("\n".join(out))
print("\n".join(out))

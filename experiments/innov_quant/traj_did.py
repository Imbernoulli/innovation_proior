# -*- coding: utf-8 -*-
"""给“先验挡住 RL 坍塌”这个结论一个检验:逐题的双重差分。
对每个 bench、每道题,算 base 血统与先验血统在 s15→s20 之间各自的变化量,再相减:
    DiD(题) = [先验血统(s20) − 先验血统(s15)] − [base 血统(s20) − base 血统(s15)]
指标:P(complete)、mean score、自相似度(同题 5 抽样的平均两两 4-gram Jaccard)。
逐题 Wilcoxon。自相似度上 DiD<0 = 先验血统的多样性掉得更少。"""
import json, glob, os, sys, itertools
import numpy as np
from collections import defaultdict
from scipy import stats
from dump2 import D, sub_of, extract_code, toks, grams, BENCHES

LIN = {
 "4B": {"base": {15:"rlv5_4b_base_s15", 20:"rlv5_4b_base_s20"},
        "ft01mix": {15:"rlv5_4b_ft01mix_a10_s15", 20:"rlv5_4b_ft01mix_a10_s20"},
        "lo32nm": {15:"rlv5_4b_lo32nm_a10_s15", 20:"rlv5_4b_lo32nm_a10_s20"}},
 "9B": {"base": {10:"rlv5_base_s10", 15:"rlv5_base_s15", 20:"rlv5_base_s20"},
        "ft01mix": {10:"rlv5_ft01mix_a10_s10", 15:"rlv5_ft01mix_a10_s15", 20:"rlv5_ft01mix_a10_s20"},
        "ft03nm": {10:"rlv5_ft03nm_a20_s10", 15:"rlv5_ft03nm_a20_s15", 20:"rlv5_ft03nm_a20_s20"},
        "lo32nm": {10:"rlv5_lo32nm_a10_s10", 15:"rlv5_lo32nm_a10_s15", 20:"rlv5_lo32nm_a10_s20"}},
}
CACHE = {}
def perprob(arm, bench):
    if (arm, bench) in CACHE: return CACHE[(arm, bench)]
    ok = {}
    for f in sorted(glob.glob(f"{D}/cc_eval_{arm}_{sub_of(bench)}/shard_*/samples.jsonl")):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("data_source") != bench or r.get("error"): continue
            m = r.get("metrics") or {}; s = m.get("score", r.get("score"))
            if s is None: continue
            ok[(str(r["ground_truth"]), int(r.get("sample_idx", -1)))] = (float(s), r.get("text") or "")
    agg = defaultdict(lambda: dict(n=0, comp=0, sc=[], codes=[]))
    for (prob, si), (sc, txt) in ok.items():
        a = agg[prob]; a["n"] += 1; a["sc"].append(sc)
        i = txt.rfind("</think>")
        if i < 0: continue
        a["comp"] += 1
        code, _ = extract_code(txt[i+8:], bench)
        if code.strip(): a["codes"].append(code)
    out = {}
    for prob, a in agg.items():
        ss = float("nan")
        if len(a["codes"]) >= 2:
            g = [grams(toks(c)[:4000]) for c in a["codes"]]
            v = [len(x & y)/len(x | y) for x, y in itertools.combinations(g, 2) if (x | y)]
            if v: ss = float(np.mean(v))
        out[prob] = dict(p_complete=a["comp"]/a["n"], score=float(np.mean(a["sc"])), self_sim=ss)
    CACHE[(arm, bench)] = out
    sys.stderr.write("loaded %s %s (%d probs)\n" % (arm, bench, len(out)))
    return out

res = []
for fam, lins in LIN.items():
    steps = sorted(set(lins["base"]) - {min(lins["base"])})     # 变化的终点 step
    s_from = min(lins["base"])
    for bench in BENCHES:
        for lin, m in lins.items():
            if lin == "base": continue
            for s_to in sorted(set(m) - {s_from}):
                A0 = perprob(m[s_from], bench); A1 = perprob(m[s_to], bench)
                B0 = perprob(lins["base"][s_from], bench); B1 = perprob(lins["base"][s_to], bench)
                probs = sorted(set(A0) & set(A1) & set(B0) & set(B1))
                for col in ["p_complete","score","self_sim"]:
                    d = []
                    for p in probs:
                        a = A1[p][col]-A0[p][col]; b = B1[p][col]-B0[p][col]
                        if np.isfinite(a) and np.isfinite(b): d.append(a-b)
                    if len(d) < 5: continue
                    d = np.array(d)
                    try: pv = stats.wilcoxon(d, zero_method="wilcox").pvalue
                    except Exception: pv = float("nan")
                    res.append(dict(fam=fam, bench=bench, lineage=lin, span="s%d->s%d" % (s_from, s_to),
                                    metric=col, n=len(d), mean=float(d.mean()),
                                    pos=int((d>0).sum()), neg=int((d<0).sum()), p=float(pv)))

out = ["# 双重差分:先验血统 vs base 血统,RL 后半程的变化差",
       "\nDiD = [先验(晚) − 先验(早)] − [base(晚) − base(早)],逐题配对 Wilcoxon。",
       "自相似度上 DiD<0 = 先验血统的多样性掉得更少(更没坍塌)。"]
for fam in ["4B","9B"]:
    for col, lab in [("p_complete","P(complete)"),("score","mean score"),("self_sim","自相似度")]:
        rs = [r for r in res if r["fam"] == fam and r["metric"] == col]
        if not rs: continue
        out.append("\n## %s — %s" % (fam, lab))
        out.append("\n| bench | 先验血统 | span | n 题 | DiD 均值 | +/− | Wilcoxon p |")
        out.append("|---|---|---|---|---|---|---|")
        for r in rs:
            out.append("| %s | %s | %s | %d | %+.4f | %d/%d | %.4f |" %
                       (r["bench"], r["lineage"], r["span"], r["n"], r["mean"], r["pos"], r["neg"], r["p"]))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_did.md"), "w").write("\n".join(out))
print("\n".join(out))

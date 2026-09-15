"""Diversity of the 5 draws (the user's question: how similar are different proposals to each other?).
Two quantities per (bench, arm, problem), over complete draws with code, 4-gram Jaccard on comment-stripped token n-grams:
  self_div  = 1 - mean pairwise similarity among the arm's OWN draws   (higher = the model proposes more different things)
  cross_div = 1 - mean pairwise similarity between the arm's draws and the control arm's draws
Paired Wilcoxon per (arm, control) over problems. Writes diversity_tables.md."""
import json, re, collections, statistics as st, math
import numpy as np
from scipy import stats
from dump2 import ARMS, MAIN, REP_OF
def strip_comments(c): return re.sub(r"//[^\n]*|/\*.*?\*/|#[^\n]*", "", c, flags=re.S)
def toks(c): return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]", c)
def grams(tk, n=4): return {tuple(tk[i:i+n]) for i in range(max(0, len(tk)-n+1))}
G = collections.defaultdict(dict)   # (bench, arm, problem) -> {sample_idx: gram set}
for l in open("samples.jsonl"):
    s = json.loads(l)
    if not s["in_common"] or not s["has_code"] or not s["complete"]: continue
    G[(s["bench"], s["arm"], s["problem"])][s["sample_idx"]] = grams(toks(strip_comments(s["code"])))
def jac(a, b): return len(a & b) / len(a | b) if (a and b) else None
def self_sim(d):
    ks = [k for k in d if d[k]]
    v = [jac(d[a], d[b]) for i, a in enumerate(ks) for b in ks[i+1:]]
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None
def cross_sim(d1, d2):
    v = [jac(a, b) for a in d1.values() for b in d2.values() if a and b]
    return st.mean(v) if v else None
L = []
def P(s=""): L.append(s)
def f(x, d=3): return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"
for bench in ("frontiercs", "frontiercs_research", "alebench"):
    P(f"# {bench}"); P()
    for fam in ("9B", "4B"):
        arms = [a for a in ARMS if ARMS[a][0] == fam]
        probs = sorted({k[2] for k in G if k[0] == bench and ARMS.get(k[1], ("",))[0] == fam})
        P(f"## {fam}"); P()
        P("### A. 自多样性:同一模型 5 次抽样之间的平均 4-gram Jaccard(越低=同一模型给出的方案越不一样)"); P()
        P("| arm | stage | n probs(>=2 draws) | mean self-sim | median | 完全不同的抽样比例(与同臂任一抽样 Jac<0.3) |"); P("|---|---|---|---|---|---|")
        SS = {}
        for a in arms:
            vals = {}; distinct = []
            for p in probs:
                d = G.get((bench, a, p), {})
                if len(d) >= 2:
                    vals[p] = self_sim(d)
                    for k, g in d.items():
                        m = max((jac(g, d[k2]) for k2 in d if k2 != k and d[k2] and g), default=None)
                        if m is not None: distinct.append(m < 0.3)
            SS[a] = vals
            v = [x for x in vals.values() if x is not None]
            P(f"| {a} | {ARMS[a][1]} | {len(v)} | {f(st.mean(v)) if v else '–'} | {f(st.median(v)) if v else '–'} | {100*st.mean(distinct):.0f}% |" if v else f"| {a} | {ARMS[a][1]} | 0 | – | – | – |")
        P()
        P("### B. 配对:该臂的自多样性 vs 对照(+ = 该臂自相似度更高,即方案更雷同)"); P()
        P("| pair | n probs | self-sim +/−/= | mean Δ | Wilcoxon p |"); P("|---|---|---|---|---|")
        pairs = [(a, ARMS[a][2]) for a in arms if ARMS[a][1] in ("sft", "rl_base", "rl_sft", "rep")] + [(a, ARMS[a][3]) for a in arms if ARMS[a][3]]
        for a, c in pairs:
            ps = [p for p in SS.get(a, {}) if p in SS.get(c, {}) and SS[a][p] is not None and SS[c][p] is not None]
            d = [SS[a][p] - SS[c][p] for p in ps]
            if len(d) < 6: P(f"| {a} − {c} | {len(d)} | – | – | – |"); continue
            pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
            wp = stats.wilcoxon(d, zero_method="wilcox").pvalue
            P(f"| {a} − {c} | {len(ps)} | {pos}−{neg}−{len(d)-pos-neg} | {f(st.mean(d))} | {f(wp)} |")
        P()
        P("### C. 跨模型相似度:该臂的抽样 vs 对照臂的抽样(平均两两 Jaccard;越低=两个模型提的方案越不一样)"); P()
        P("| pair | n probs | mean cross-sim | arm 自相似 | ctrl 自相似 | 跨模型 − 两者自相似均值(>0 说明两模型比各自内部还像) |"); P("|---|---|---|---|---|---|")
        for a, c in pairs:
            ps = [p for p in probs if G.get((bench, a, p)) and G.get((bench, c, p))]
            xs = [cross_sim(G[(bench, a, p)], G[(bench, c, p)]) for p in ps]
            xs = [x for x in xs if x is not None]
            sa = [SS[a][p] for p in ps if SS.get(a, {}).get(p) is not None]; sc = [SS[c][p] for p in ps if SS.get(c, {}).get(p) is not None]
            if not xs: continue
            base = (st.mean(sa) + st.mean(sc)) / 2 if sa and sc else None
            P(f"| {a} − {c} | {len(xs)} | {f(st.mean(xs))} | {f(st.mean(sa)) if sa else '–'} | {f(st.mean(sc)) if sc else '–'} | {f(st.mean(xs)-base) if base else '–'} |")
        P()
open("diversity_tables.md", "w").write("\n".join(L)); print("\n".join(L))

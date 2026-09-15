"""Statistics over oracle_metrics.csv (+ metrics.csv): closeness of our complete FCS solutions to the frontier-model pool
(and to the two human references that exist). Writes oracle_tables.md."""
import csv, json, os, glob, re, collections, statistics as st, math
import numpy as np
from scipy import stats
from multiprocessing import Pool
from dump2 import ARMS, MAIN, REP_OF
from oracle_sim2 import load_pool, toks, grams, strip_comments
OUT = os.path.dirname(os.path.abspath(__file__)); L = []
def P(s=""): L.append(s)
def table(h, rws):
    P("| " + " | ".join(h) + " |"); P("|" + "|".join("---" for _ in h) + "|")
    for r in rws: P("| " + " | ".join(str(x) for x in r) + " |")
    P()
def f(x, d=3): return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
def fl(x):
    try: return float(x)
    except: return None
def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None
def med(xs): xs = [x for x in xs if x is not None]; return st.median(xs) if xs else None
rows = list(csv.DictReader(open(f"{OUT}/oracle_metrics.csv")))
for r in rows:
    for k in ("score", "sim_human", "jac_frontier", "sim_frontier"): r[k] = fl(r[k])
    r["sample_idx"] = int(r["sample_idx"]); r["in_common"] = r["in_common"] == "True"; r["complete"] = r["complete"] == "True"
rows = [r for r in rows if r["in_common"] and r["complete"] and r["jac_frontier"] is not None and r["sim_frontier"] is not None]
PASS = lambda s: s >= 100 - 1e-9
BY = collections.defaultdict(list)
for r in rows: BY[(r["bench"], r["arm"])].append(r)

def frontier_self(args):
    bench, prob = args
    hu, fr = load_pool(bench, prob)
    G = [(m, grams(toks(strip_comments(c)))) for m, c in fr]
    out = []
    for i, (m, g) in enumerate(G):
        if not g: continue
        best = 0.0
        for j, (m2, g2) in enumerate(G):
            if m2 == m or not g2: continue
            best = max(best, len(g & g2) / len(g | g2))
        out.append(best)
    return bench, prob, out

if __name__ == "__main__":
    for bench in ("frontiercs", "frontiercs_research"):
        P(f"# {bench}: 与官方 frontier-model 解池的相似度(complete 且有代码的 common 样本)"); P()
        probs_all = sorted({r["problem"] for r in rows if r["bench"] == bench})
        # calibration: frontier model vs the other frontier models (same metric)
        with Pool(int(os.environ.get("NPROC", "12"))) as pool: cal = list(pool.imap_unordered(frontier_self, [(bench, p) for p in probs_all], chunksize=4))
        calv = [x for _, _, v in cal for x in v]; calp = {p: mean(v) for _, p, v in cal}
        P(f"标尺:frontier 模型的解与**其他** frontier 模型的解的最大 4-gram Jaccard(留一模型):median {f(med(calv))}, q25 {f(np.quantile(calv, .25))}, q75 {f(np.quantile(calv, .75))}, n={len(calv)}(题数 {len(probs_all)})"); P()
        for fam in ("9B", "4B"):
            arms = [a for a in ARMS if ARMS[a][0] == fam]
            P(f"## {fam}"); P()
            rws = []
            for a in arms:
                rs = BY[(bench, a)]
                if not rs: continue
                nf = collections.Counter(r["nearest_frontier"] for r in rs).most_common(3)
                rws.append([a, ARMS[a][1], len(rs), f(med([r["jac_frontier"] for r in rs])), f(med([r["sim_frontier"] for r in rs])), f"{100*mean([r['jac_frontier'] > 0.3 for r in rs]):.0f}%", f"{100*mean([r['sim_frontier'] > 0.5 for r in rs]):.0f}%",
                            f(mean([r["jac_frontier"] - calp.get(r["problem"], 0) for r in rs])), ", ".join(f"{m}:{c}" for m, c in nf)])
            table(["arm", "stage", "n", "med jac_frontier", "med sim_frontier", "jac>0.3", "sim>0.5", "mean(jac − frontier self-jac of same problem)", "nearest frontier model (top3)"], rws)
            # paired: per-problem mean jac_frontier, arm vs control / start / rep
            P("### 配对(按题均值):该臂比对照更像/更不像 frontier 池?"); P()
            rws = []
            pairs = [(a, ARMS[a][2]) for a in arms if ARMS[a][1] in ("sft", "rl_base", "rl_sft", "rep")] + [(a, ARMS[a][3]) for a in arms if ARMS[a][3]]
            pm = {a: collections.defaultdict(list) for a in arms}
            for a in arms:
                for r in BY[(bench, a)]: pm[a][r["problem"]].append(r)
            for a, c in pairs:
                ps = [p for p in pm[a] if p in pm[c]]
                d = [mean([r["jac_frontier"] for r in pm[a][p]]) - mean([r["jac_frontier"] for r in pm[c][p]]) for p in ps]
                pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
                wp = stats.wilcoxon(d, zero_method="wilcox").pvalue if pos + neg >= 6 else None
                # wins: where arm best > ctrl best, is the arm's best sample closer to the frontier pool than ctrl's best?
                aw = [p for p in ps if max(r["score"] for r in pm[a][p]) > max(r["score"] for r in pm[c][p])]
                closer = sum(1 for p in aw if max(pm[a][p], key=lambda r: r["score"])["jac_frontier"] > max(pm[c][p], key=lambda r: r["score"])["jac_frontier"])
                cw = [p for p in ps if max(r["score"] for r in pm[c][p]) > max(r["score"] for r in pm[a][p])]
                closer_c = sum(1 for p in cw if max(pm[c][p], key=lambda r: r["score"])["jac_frontier"] > max(pm[a][p], key=lambda r: r["score"])["jac_frontier"])
                rws.append([f"{a} − {c}", len(ps), f"{pos}−{neg}−{len(d)-pos-neg}", f(mean(d)), f(wp), f"{closer}/{len(aw)}", f"{closer_c}/{len(cw)}"])
            table(["pair", "n probs", "jac_frontier +/−/=", "mean Δ", "Wilcoxon p", "arm wins where arm's best is closer to frontier pool", "ctrl wins where ctrl's best is closer"], rws)
            # within-problem association
            P("### 题内关联:像 frontier 池 ↔ 分数(main 臂样本,题内排秩)"); P()
            rws = []
            for key, name in (("jac_frontier", "4-gram Jaccard to nearest frontier"), ("sim_frontier", "token ratio to nearest frontier")):
                byp = collections.defaultdict(list)
                for a in arms:
                    if ARMS[a][1] == "rep": continue
                    for r in BY[(bench, a)]: byp[r["problem"]].append(r)
                xs, ys = [], []; tert = collections.defaultdict(collections.Counter)
                for p, rs in byp.items():
                    if len(rs) < 6: continue
                    xs += list(stats.rankdata([r[key] for r in rs]) / len(rs)); ys += list(stats.rankdata([r["score"] for r in rs]) / len(rs))
                    q = np.quantile([r[key] for r in rs], [1/3, 2/3])
                    for r in rs:
                        b = "low" if r[key] <= q[0] else ("mid" if r[key] <= q[1] else "high")
                        tert[b]["n"] += 1; tert[b]["pass"] += PASS(r["score"]); tert[b]["s"] += r["score"]
                rho, pv = stats.spearmanr(xs, ys)
                rws.append([name, len(xs), f(rho), f(pv, 4)] + [f"{f(tert[b]['s']/max(1,tert[b]['n']),2)} / {100*tert[b]['pass']/max(1,tert[b]['n']):.0f}%" for b in ("low", "mid", "high")])
            table(["property", "n", "Spearman rho vs score", "p", "low tercile mean/pass", "mid", "high"], rws)
        # human refs anecdotes
        hr = [r for r in rows if r["bench"] == bench and r["sim_human"] is not None]
        if hr:
            P("## 有人类参考解的题(仅此几题)"); P()
            rws = []
            for (p, a), rs in sorted(collections.defaultdict(list, {(r["problem"], r["arm"]): [x for x in hr if x["problem"] == r["problem"] and x["arm"] == r["arm"]] for r in hr}).items()):
                rws.append([p, a, len(rs), f(max(r["sim_human"] for r in rs)), f(mean([r["score"] for r in rs]), 1), f(max(r["score"] for r in rs), 1)])
            table(["problem", "arm", "n", "max sim to human ref", "mean score", "best score"], rws)
    open(f"{OUT}/oracle_tables.md", "w").write("\n".join(L)); print("\n".join(L))

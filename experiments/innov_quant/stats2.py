"""Stage-1 statistics v2 over metrics.csv from dump2.py.
Adds: completion/truncation decomposition, same-protocol replicate noise floor + same-model similarity floor,
judge-topology validity for ALE pairs, count of truncated-but-scored samples. Writes report_tables.md + wins.json."""
import csv, json, collections, statistics as st, math, os
import numpy as np
from scipy import stats
from dump2 import ARMS, MAIN, REP_OF, ORIG_REP
OUT = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(f"{OUT}/metrics.csv")))
cov = json.load(open(f"{OUT}/coverage.json"))
def fl(x):
    try: return float(x)
    except: return None
for r in rows:
    for k in ("score","score_unbounded","performance","completion_tokens","reasoning_len","final_len","loc","n_tokens","n_loops","n_if","n_func","n_numlit","n_ident","sim_ctrl","sim_start","sim_self","sim_rep","jac_pool"): r[k] = fl(r.get(k))
    for k in ("in_common","in_full","has_code","complete","trunc"): r[k] = r[k] == "True"
    r["nov_ctrl"] = (1 - r["sim_ctrl"]) if r["sim_ctrl"] is not None else None
    r["nov_pool"] = (1 - r["jac_pool"]) if r["jac_pool"] is not None else None
PASS = {"frontiercs": lambda s: s >= 100 - 1e-9, "frontiercs_research": lambda s: s >= 100 - 1e-9, "alebench": lambda s: s > 0}
L = []
def P(s=""): L.append(s)
def table(h, rws):
    P("| " + " | ".join(h) + " |"); P("|" + "|".join("---" for _ in h) + "|")
    for r in rws: P("| " + " | ".join(str(x) for x in r) + " |")
    P()
def f(x, d=3): return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
def med(xs): xs = [x for x in xs if x is not None]; return st.median(xs) if xs else None
def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None
def speed(bench, arm):
    js = cov.get(f"{bench}|{arm}", {}).get("judge", []); sp = [j["speed"] for j in js if j.get("speed")]
    return (mean(sp), sorted({j["partition"] for j in js if j.get("partition")})) if sp else (None, [])
def topo_ok(bench, a, c):
    if bench != "alebench": return True
    sa, sc = speed(bench, a)[0], speed(bench, c)[0]
    return sa is not None and sc is not None and abs(sa - sc) < 0.1
BYB = collections.defaultdict(list)
for r in rows: BYB[(r["bench"], r["arm"])].append(r)
def per_problem(bench, arm):
    d = collections.defaultdict(list)
    for r in BYB[(bench, arm)]:
        if r["in_common"]: d[r["problem"]].append(r)
    out = {}
    for p, rs in d.items():
        sc = [r["score"] for r in rs]; best = max(rs, key=lambda r: r["score"]); comp = [r for r in rs if r["complete"]]
        out[p] = dict(mean=mean(sc), best=max(sc), worst=min(sc), pass_=any(PASS[bench](s) for s in sc), n=len(rs), best_row=best, rows=rs,
                      p_complete=len(comp) / len(rs), mean_complete=mean([r["score"] for r in comp]) if comp else None)
    return out
def sign_w(d):
    d = [x for x in d if x is not None]
    pos = sum(x > 0 for x in d); neg = sum(x < 0 for x in d)
    wp = stats.wilcoxon(d, zero_method="wilcox").pvalue if pos + neg >= 6 else None
    sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else None
    return pos, neg, len(d) - pos - neg, wp, sp
def pstr(t): return f"{t[0]}−{t[1]}−{t[2]}"

wins = {}; TAU = 0.5
for bench in ("frontiercs", "frontiercs_research", "alebench"):
    P(f"# {bench}"); P()
    for fam in ("9B", "4B"):
        mains = [a for a in MAIN if ARMS[a][0] == fam]; reps = [a for a in REP_OF if ARMS[a][0] == fam]
        arms = mains + reps
        pp = {a: per_problem(bench, a) for a in arms}
        probs = sorted(set.intersection(*(set(pp[a]) for a in mains)))
        P(f"## {fam}(common problems n={len(probs)};replicate arms use the same problem set where available)"); P()
        # A. per-arm summary (+ judge topology)
        P("### A. 每臂概览(仅 common 题;code 指标只算 complete 且有代码的样本)"); P()
        rws = []
        for a in arms:
            ps_ = [p for p in probs if p in pp[a]]
            rs = [r for r in BYB[(bench, a)] if r["in_common"] and r["problem"] in set(ps_)]
            rc = [r for r in rs if r["has_code"]]
            sp, parts = speed(bench, a)
            rws.append([a, ARMS[a][1], len(ps_), f(mean([pp[a][p]["mean"] for p in ps_]), 2), f(mean([pp[a][p]["best"] for p in ps_]), 2), f"{100*mean([pp[a][p]['pass_'] for p in ps_]):.0f}%",
                        f"{100*mean([r['complete'] for r in rs]):.0f}%", f"{100*mean([r['trunc'] for r in rs]):.0f}%", f(mean([r["score"] for r in rs if r["complete"]]), 2),
                        sum(1 for r in rs if r["trunc"] and r["score"] > 0),
                        f(med([r["loc"] for r in rc]), 0), f(med([r["reasoning_len"] for r in rs]), 0), f(med([r["completion_tokens"] for r in rs]), 0),
                        f(med([r["sim_ctrl"] for r in rc])), f(med([r["sim_self"] for r in rc])), f(med([r["jac_pool"] for r in rc])),
                        f"{100*mean([(r['nov_ctrl'] or 0) > TAU for r in rc if r['nov_ctrl'] is not None]):.0f}%" if rc else "–",
                        "/".join(parts) + (f" {sp:.2f}" if sp else "")])
        table(["arm","stage","n probs","mean@5","best@5","pass@5","complete","trunc","E[score|complete]","trunc&score>0","med LOC","med reasoning chars","med compl. tok","med sim_ctrl","med sim_self","med jac_pool",f"nov_ctrl>{TAU}","judge partition/speed"], rws)
        # B. paired comparisons
        pairs = [(a, ARMS[a][2]) for a in mains if ARMS[a][1] in ("sft", "rl_base", "rl_sft")] + [(a, ARMS[a][3]) for a in mains if ARMS[a][3]]
        # replicate pairs (noise floor) and, for ALE topology-confounded pairs, replicate substitutes
        rep_pairs = [(a, REP_OF[a]) for a in reps if bench in ("frontiercs", "frontiercs_research", "alebench")]
        subs = []
        for a, c in pairs:
            if not topo_ok(bench, a, c):
                for a2, c2 in ((a, ORIG_REP.get(c)), (ORIG_REP.get(a), c), (ORIG_REP.get(a), ORIG_REP.get(c))):
                    if a2 and c2 and a2 in pp and c2 in pp and topo_ok(bench, a2, c2): subs.append((a2, c2))
        P("### B. 配对(按题):分数差、完成率差、条件分差、赢题的新颖比例"); P()
        P(f"novel = 该臂在该题最好样本与对照在该题全部样本的最大 token 相似度 < {1-TAU};赢 = best@5 严格更高。⚠ = ALE 判题节点 speedFactor 相差 ≥0.1(不同拓扑),该对不可比。E[score|complete] 差只在两边都至少有 1 个 complete 样本的题上算。"); P()
        rws = []
        def pair_row(a, c, tag=""):
            ppa, ppc = pp[a], pp[c]; ps_ = [p for p in probs if p in ppa and p in ppc]
            d_mean = [ppa[p]["mean"] - ppc[p]["mean"] for p in ps_]; d_best = [ppa[p]["best"] - ppc[p]["best"] for p in ps_]
            d_pass = [int(ppa[p]["pass_"]) - int(ppc[p]["pass_"]) for p in ps_]
            d_comp = [ppa[p]["p_complete"] - ppc[p]["p_complete"] for p in ps_]
            d_cond = [ppa[p]["mean_complete"] - ppc[p]["mean_complete"] for p in ps_ if ppa[p]["mean_complete"] is not None and ppc[p]["mean_complete"] is not None]
            pm, pb, ps, pc, pd = sign_w(d_mean), sign_w(d_best), sign_w(d_pass), sign_w(d_comp), sign_w(d_cond)
            a_wins = [p for p in ps_ if ppa[p]["best"] > ppc[p]["best"]]; c_wins = [p for p in ps_ if ppc[p]["best"] > ppa[p]["best"]]
            simkey = "sim_ctrl" if c == ARMS[a][2] else ("sim_start" if c == ARMS[a][3] else None)
            a_win_nov = [p for p in a_wins if simkey and ppa[p]["best_row"].get(simkey) is not None and 1 - ppa[p]["best_row"][simkey] > TAU]
            # symmetric: control's wins that are novel vs the arm (control best row sim to arm's samples ~ use arm-side max: recompute cheaply)
            ok = "" if topo_ok(bench, a, c) else "⚠ "
            rws.append([f"{ok}{tag}{a} − {c}", len(ps_), pstr(pm), f(mean(d_mean), 2), f(pm[3]), f(pm[4]), pstr(pb), f(pb[3]), f"{ps[0]}−{ps[1]}", f(ps[4]),
                        pstr(pc), f(mean(d_comp), 3), f(pc[3]), f"{pstr(pd)} (n={len(d_cond)})", f(mean(d_cond), 2), f(pd[3]),
                        (f"{len(a_win_nov)}/{len(a_wins)}" if simkey else f"–/{len(a_wins)}"), (f(mean([ppa[p]['best_row'].get(simkey) for p in a_wins]), 2) if simkey else "–")])
            wins[f"{bench}|{a}|{c}"] = dict(arm_wins=a_wins, ctrl_wins=c_wins, arm_wins_novel=a_win_nov, probs=ps_, topo_ok=topo_ok(bench, a, c), kind=tag.strip("[] ") or "main")
        for a, c in pairs: pair_row(a, c)
        for a, c in subs: pair_row(a, c, "[sub] ")
        for a, c in rep_pairs: pair_row(a, c, "[rep] ")
        table(["pair","n","mean@5 +/−/=","mean Δ","Wilcoxon p","sign p","best@5 +/−/=","Wilcoxon p","pass@5 +/−","sign p","P(complete) +/−/=","mean Δ","Wilcoxon p","E[score|complete] +/−/=","mean Δ","Wilcoxon p","arm wins novel / arm wins","mean sim(best,ctrl) on wins"], rws)
        # B2. similarity floor: same-model cross-run vs across-arm
        P("### B2. 相似度标尺:同模型两次独立跑(rep) vs 不同臂;每样本 = 与对照臂同题样本的最大 token 相似度"); P()
        rws = []
        for a in arms:
            rc = [r for r in BYB[(bench, a)] if r["in_common"] and r["has_code"] and r["sim_ctrl"] is not None]
            if not rc: continue
            v = [r["sim_ctrl"] for r in rc]
            rws.append([a, ARMS[a][1], ARMS[a][2], len(v), f(np.quantile(v, .25)), f(st.median(v)), f(np.quantile(v, .75)), f"{100*mean([x > 0.5 for x in v]):.0f}%", f"{100*mean([x > 0.8 for x in v]):.0f}%"])
        table(["arm","stage","vs","n samples","q25","median","q75","sim>0.5","sim>0.8"], rws)
        # C. within-problem association
        P("### C. 题内关联(全族 main 臂 complete 样本在题内排秩后合并):属性 ↔ 分数"); P()
        rws = []
        for key, name in [("nov_ctrl", "novelty vs control"), ("nov_pool", "novelty vs pool (1−jac_pool)"), ("loc", "LOC"), ("reasoning_len", "reasoning chars"), ("completion_tokens", "compl. tokens"), ("n_numlit", "#num literals")]:
            xs, ys = [], []
            byp = collections.defaultdict(list)
            for a in mains:
                for r in BYB[(bench, a)]:
                    if r["in_common"] and r["has_code"] and r.get(key) is not None: byp[r["problem"]].append(r)
            for p in probs:
                rs = byp.get(p, [])
                if len(rs) < 6: continue
                rk = stats.rankdata([r[key] for r in rs]) / len(rs); sk = stats.rankdata([r["score"] for r in rs]) / len(rs)
                xs += list(rk); ys += list(sk)
            if len(xs) > 10:
                rho, pv = stats.spearmanr(xs, ys); rws.append([name, len(xs), f(rho), f(pv, 4)])
        table(["property","n samples","Spearman rho vs score","p"], rws)
        # D. novelty tercile
        P("### D. 题内 novelty(vs control)三分位 × 结果(main 非 base 臂,complete 样本)"); P()
        tert = collections.defaultdict(collections.Counter)
        byp = collections.defaultdict(list)
        for a in mains:
            if ARMS[a][1] == "base": continue
            for r in BYB[(bench, a)]:
                if r["in_common"] and r["has_code"] and r["nov_ctrl"] is not None: byp[r["problem"]].append(r)
        for p in probs:
            rs = byp.get(p, [])
            if len(rs) < 6: continue
            q = np.quantile([r["nov_ctrl"] for r in rs], [1/3, 2/3])
            for r in rs:
                b = "low" if r["nov_ctrl"] <= q[0] else ("mid" if r["nov_ctrl"] <= q[1] else "high")
                tert[b]["n"] += 1; tert[b]["pass"] += PASS[bench](r["score"]); tert[b]["score_sum"] += r["score"]
        table(["novelty tercile","n","pass rate","mean score"], [[b, tert[b]["n"], f"{100*tert[b]['pass']/max(1,tert[b]['n']):.0f}%", f(tert[b]["score_sum"]/max(1,tert[b]["n"]), 2)] for b in ("low","mid","high")])
open(f"{OUT}/report_tables.md", "w").write("\n".join(L)); json.dump(wins, open(f"{OUT}/wins.json", "w"), indent=1)
print("\n".join(L))

"""Paired, per-problem version of the cross-domain-vocabulary indicator (the aggregate table in
experiments/scripts/agg/crossdomain_lex.py is unpaired and dominated by "simulated annealing", which is the
standard tool for ALE heuristic problems, so it is not evidence of cross-field transfer).
Per (bench, arm, problem): share of the arm's draws whose FULL TEXT mentions >=1 cross-domain term.
Two vocabularies: ALL terms, and ALL minus simulated annealing. Paired Wilcoxon arm vs control over problems."""
import re, collections, statistics as st, json
from scipy import stats
from dump2 import ARMS, load, BENCHES
TERMS = {
 "optimal stopping / secretary": r"optimal stopping|secretary problem", "martingale / stopping time": r"martingale|stopping time",
 "Lyapunov / potential function": r"lyapunov|potential function", "Hamiltonian (physics)": r"hamiltonian (mechanic|system|structure|dynamic|energy)",
 "simulated annealing": r"simulated annealing", "free energy / partition function": r"free energy|partition function",
 "renormalization / mean field": r"renormali|mean[- ]field", "phase transition / percolation": r"phase transition|percolation",
 "queueing / renewal": r"queueing theory|queuing theory|renewal (process|theory)|little's law",
 "bandit / Thompson": r"multi[- ]armed bandit|thompson sampling|upper confidence bound", "Bayesian posterior": r"bayesian posterior|posterior distribution",
 "KKT / duality": r"\bkkt\b|lagrangian dual|convex dualit|primal[- ]dual", "submodular / matroid": r"submodular|matroid",
 "concentration bound": r"chernoff|hoeffding|concentration (bound|inequalit)",
 "mutual information": r"mutual information|information bottleneck|rate[- ]distortion", "optimal transport": r"optimal transport|wasserstein",
 "Kalman / observer / PID": r"kalman filter|state observer|pid controller", "spectral gap / mixing time": r"spectral gap|mixing time",
 "generating function / FFT": r"generating function|fast fourier|number[- ]theoretic transform", "fixed point / contraction": r"contraction mapping|banach fixed",
 "dimensional analysis": r"dimensional analysis|conservation law", "extreme value theory": r"extreme value theor"}
UNION = re.compile("|".join(f"(?:{v})" for v in TERMS.values()), re.I)
NOSA = re.compile("|".join(f"(?:{v})" for k, v in TERMS.items() if k != "simulated annealing"), re.I)
hits = collections.defaultdict(lambda: [0, 0, 0])   # (bench, arm, problem) -> [n, hit_all, hit_nosa]
for bench in BENCHES:
    for arm in ARMS:
        ok, _ = load(arm, bench)
        for (prob, si), v in ok.items():
            t = v["text"]; h = hits[(bench, arm, prob)]
            h[0] += 1; h[1] += bool(UNION.search(t)); h[2] += bool(NOSA.search(t))
L = []
for bench in BENCHES:
    L.append(f"# {bench}\n")
    for fam in ("9B", "4B"):
        arms = [a for a in ARMS if ARMS[a][0] == fam]
        L.append(f"## {fam}\n")
        rate = {}
        for a in arms:
            rate[a] = {p: (h[1] / h[0], h[2] / h[0]) for (b, ar, p), h in hits.items() if b == bench and ar == a and h[0]}
            v = list(rate[a].values())
            if v: L.append(f"- `{a}` ({ARMS[a][1]}): 含跨域词的抽样比例 {100*st.mean(x[0] for x in v):.1f}% / 去掉 simulated annealing 后 {100*st.mean(x[1] for x in v):.1f}%(n={len(v)} 题)")
        L.append("")
        L.append("| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |"); L.append("|---|---|---|---|---|---|---|---|")
        pairs = [(a, ARMS[a][2]) for a in arms if ARMS[a][1] in ("sft", "rl_base", "rl_sft", "rep")] + [(a, ARMS[a][3]) for a in arms if ARMS[a][3]]
        for a, c in pairs:
            ps = [p for p in rate[a] if p in rate[c]]
            out = []
            for i in (0, 1):
                d = [rate[a][p][i] - rate[c][p][i] for p in ps]
                pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
                wp = stats.wilcoxon(d, zero_method="wilcox").pvalue if pos + neg >= 6 else float("nan")
                out.append(f"{pos}−{neg}−{len(d)-pos-neg} | {st.mean(d):+.3f} | {wp:.3f}")
            L.append(f"| {a} − {c} | {len(ps)} | " + " | ".join(out) + " |")
        L.append("")
open("crossdomain_tables.md", "w").write("\n".join(L)); print("\n".join(L))

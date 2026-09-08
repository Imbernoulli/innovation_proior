"""Unbiased cross-domain-vocabulary count over the FULL eval corpus.

The case bundles in scratchpad/case are score-selected, so counts taken there are
biased upwards for whichever arm won. This walks every sample of every arm on the
two FrontierCS tracks plus liveidea_gen instead, and reports, per arm, the share of
DRAWS that mention each technique at least once (share, not raw hits, so a single
verbose draw cannot carry a term).
"""
import json, glob, re, os, collections, statistics

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
ARMS = ["base9b_v2c", "ft01mix_a10", "ft03nm_a20", "lo32nm_a10",
        "rlv5_base_s20", "rlv5_ft01mix_a10_s20", "rlv5_ft03nm_a20_s20", "rlv5_lo32nm_a10_s20"]
# Terms whose home field is NOT computer science, or is a distant CS subfield.
# "annealing" is deliberately split: CosineAnnealingLR is PyTorch boilerplate, so
# only the standalone physics phrase counts.
TERMS = {
    "optimal stopping / secretary": r"optimal stopping|secretary problem",
    "martingale / stopping time": r"martingale|stopping time",
    "Lyapunov / potential function": r"lyapunov|potential function",
    "Hamiltonian (physics sense)": r"hamiltonian (mechanic|system|structure|dynamic|energy)",
    "simulated annealing": r"simulated annealing",
    "free energy / partition function": r"free energy|partition function",
    "renormalization / mean field": r"renormali|mean[- ]field",
    "phase transition / percolation": r"phase transition|percolation",
    "queueing / renewal / Little's law": r"queueing theory|queuing theory|renewal (process|theory)|little's law",
    "bandit / Thompson sampling": r"multi[- ]armed bandit|thompson sampling|upper confidence bound",
    "Bayesian posterior": r"bayesian posterior|posterior distribution",
    "KKT / duality / Lagrangian": r"\bkkt\b|lagrangian dual|convex dualit|primal[- ]dual",
    "submodular / matroid": r"submodular|matroid",
    "concentration bound": r"chernoff|hoeffding|concentration (bound|inequalit)",
    "mutual information / info bottleneck": r"mutual information|information bottleneck|rate[- ]distortion",
    "optimal transport / Wasserstein": r"optimal transport|wasserstein",
    "Kalman / observer / PID": r"kalman filter|state observer|pid controller",
    "spectral gap / mixing time": r"spectral gap|mixing time",
    "generating function / FFT / NTT": r"generating function|fast fourier|number[- ]theoretic transform",
    "fixed point / contraction mapping": r"contraction mapping|banach fixed",
    "dimensional analysis / conservation law": r"dimensional analysis|conservation law",
    "extreme value theory": r"extreme value theor",
}
PAT = {k: re.compile(v, re.I) for k, v in TERMS.items()}
UNION = re.compile("|".join(f"(?:{v})" for v in TERMS.values()), re.I)


def draws(arm):
    for sub in ["thinking_32k_both_vllm", "research_thinking_32k_vllm"]:
        for f in glob.glob(f"{D}/outputs/cc_eval_{arm}_{sub}/shard_*/samples.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("text"):
                    yield r["text"]
    f = f"{D}/outputs/cc_gen_{arm}/samples.jsonl"
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            if r["task"] == "liveidea_gen" and r.get("text"):
                yield r["text"]


def main():
    hit = {a: collections.Counter() for a in ARMS}
    n = {}
    for a in ARMS:
        c = 0
        for t in draws(a):
            c += 1
            if not UNION.search(t):
                continue
            for k, p in PAT.items():
                if p.search(t):
                    hit[a][k] += 1
        n[a] = c
    print("| 技术词(母学科在本题领域之外) | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
    print("|---" * (len(ARMS) + 1) + "|")
    for k in TERMS:
        row = [f"{100.0*hit[a][k]/n[a]:.2f}%" for a in ARMS]
        print(f"| {k} | " + " | ".join(row) + " |")
    print("| **抽样总数** | " + " | ".join(str(n[a]) for a in ARMS) + " |")
    tot = {a: sum(hit[a].values()) for a in ARMS}
    print("| **全部词合计命中率** | " + " | ".join(f"{100.0*tot[a]/n[a]:.2f}%" for a in ARMS) + " |")


main()

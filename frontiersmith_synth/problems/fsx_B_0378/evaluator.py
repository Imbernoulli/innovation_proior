#!/usr/bin/env python3
# FROZEN evaluator for fsx_B_0378 -- "Asteroid-Belt Buffer Allocation across a Refining Tree"
# Family: constrained-OR (multi-echelon safety-stock / newsvendor under a budget + service constraint).
# Deterministic, seeded. The candidate is UNTRUSTED and runs OS-isolated via isorun; it only ever
# sees inst["public"]. Held-out demand scenarios live in inst["hidden"] (parent process only).
import sys, json, random, isorun
from statistics import NormalDist

N = NormalDist(0, 1)

# -------- instance distribution (seeded, deterministic) --------
# A rooted refining tree of processing stations on a mining flotilla. Each node i buffers a
# common reagent to meet uncertain per-shift demand D_i. A parent station's leftover reagent can
# be routed DOWN to cover its immediate children's shortfalls (single-level risk pooling) at a
# per-unit transfer cost. Objective: minimize expected (holding + transfer + shortage) cost.
_SPECS = [
    # (seed, n, cv_lo, cv_hi, ratio_lo, ratio_hi, budget_factor)
    (7001, 12, 0.20, 0.40, 4.0,  9.0, 1.10),
    (7002, 13, 0.25, 0.45, 5.0, 11.0, 1.00),
    (7003, 14, 0.20, 0.35, 6.0, 12.0, 0.95),
    (7004, 15, 0.30, 0.50, 4.0,  8.0, 1.05),
    (7005, 12, 0.25, 0.45, 7.0, 14.0, 0.85),  # harder: costly shortage, tight budget
    (7006, 16, 0.20, 0.40, 5.0, 10.0, 0.90),
    (7007, 14, 0.30, 0.50, 8.0, 15.0, 0.80),  # hardest: high volatility + tightest budget
    (7008, 13, 0.20, 0.35, 4.0,  9.0, 1.15),
]
_N_SCEN = 200


def make_instances():
    out = []
    for (seed, n, cvlo, cvhi, rlo, rhi, bfac) in _SPECS:
        rng = random.Random(seed)
        parent = [-1] + [rng.randint(0, i - 1) for i in range(1, n)]
        mean = [rng.uniform(20.0, 80.0) for _ in range(n)]
        std = [mean[i] * rng.uniform(cvlo, cvhi) for i in range(n)]
        h = [rng.uniform(1.0, 3.0) for _ in range(n)]            # holding cost / unit leftover
        p = [h[i] * rng.uniform(rlo, rhi) for i in range(n)]      # shortage penalty / unit unmet
        t = [h[i] * rng.uniform(0.3, 0.8) for i in range(n)]      # transfer (pooling) cost / unit
        budget = sum(mean) + bfac * sum(std)
        # held-out realized demand scenarios (hidden from the candidate)
        srng = random.Random(seed * 13 + 5)
        scen = [[max(0.0, srng.gauss(mean[i], std[i])) for i in range(n)] for _ in range(_N_SCEN)]
        pub = {"n": n, "parent": parent, "h": h, "p": p, "t": t,
               "mean": mean, "std": std, "budget": budget, "n_scenarios": _N_SCEN}
        # service floor is derived from the trivial equal-split allocation so it is always feasible
        _, fill = _simulate(pub, scen, _equal_split(pub))
        pub["service_target"] = round(fill - 0.02, 4)
        out.append({"public": pub, "hidden": {"scenarios": scen}})
    return out


def _equal_split(pub):
    return [pub["budget"] / pub["n"]] * pub["n"]


# -------- deterministic settlement / cost model --------
def _simulate(pub, scen, q):
    n = pub["n"]; parent = pub["parent"]; h = pub["h"]; p = pub["p"]; t = pub["t"]
    children = [[] for _ in range(n)]
    for i, pa in enumerate(parent):
        if pa >= 0:
            children[pa].append(i)
    tot_cost = 0.0; tot_dem = 0.0; tot_unmet = 0.0
    for D in scen:
        surplus = [q[i] - D[i] if q[i] > D[i] else 0.0 for i in range(n)]
        deficit = [D[i] - q[i] if D[i] > q[i] else 0.0 for i in range(n)]
        covered = [0.0] * n
        cost = 0.0
        for i in range(n):
            ch = children[i]
            if not ch:
                cost += h[i] * surplus[i]
                continue
            tot_def = 0.0
            for c in ch:
                tot_def += deficit[c]
            transfer = surplus[i] if surplus[i] < tot_def else tot_def
            if tot_def > 1e-12 and transfer > 0.0:
                for c in ch:
                    covered[c] += transfer * deficit[c] / tot_def
            cost += h[i] * (surplus[i] - transfer) + t[i] * transfer
        for i in range(n):
            resid = deficit[i] - covered[i]
            if resid < 0.0:
                resid = 0.0
            cost += p[i] * resid
            tot_unmet += resid
            tot_dem += D[i]
        tot_cost += cost
    ns = len(scen)
    return tot_cost / ns, 1.0 - tot_unmet / max(tot_dem, 1e-12)


def baseline(inst):
    # cost of the trivial equal-split buffer allocation (evaluator computes this itself)
    pub = inst["public"]
    c, _ = _simulate(pub, inst["hidden"]["scenarios"], _equal_split(pub))
    return c


def score(inst, ans):
    pub = inst["public"]; n = pub["n"]
    if not isinstance(ans, dict) or "stock" not in ans:
        return False, 0.0
    q = ans["stock"]
    if not isinstance(q, list) or len(q) != n:
        return False, 0.0
    tot = 0.0
    for v in q:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, 0.0
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        if v < -1e-9:
            return False, 0.0
        tot += v
    if tot > pub["budget"] * (1 + 1e-9):        # over the buffer budget -> infeasible
        return False, 0.0
    obj, fill = _simulate(pub, inst["hidden"]["scenarios"], [max(0.0, v) for v in q])
    if fill < pub["service_target"] - 1e-9:      # below the service floor -> infeasible
        return False, 0.0
    if obj != obj or obj in (float("inf"), float("-inf")):
        return False, 0.0
    return True, obj


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))   # minimization: trivial construction -> ~0.1
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()

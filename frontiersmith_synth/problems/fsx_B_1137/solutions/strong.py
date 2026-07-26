# TIER: strong
# INSIGHT: the line-wide cost is dominated by the tooling STATES the chosen order
# irreversibly commits every station into, not by any single job's own attribute (SPT
# ignores this entirely). Two forces pull in opposite directions across the M stations:
#   - a station's warm-up window rewards long, unbroken same-type runs (memory-based
#     setup: cost falls the more of the last H_m jobs at that station share the current
#     type), favoring BIG blocks;
#   - a station's fatigue limit punishes a run once it exceeds a threshold R_m, favoring
#     SMALL blocks (or periodic breaks).
# Because the SAME permutation feeds every station (one-way, no re-sequencing), there is
# no per-station fix after the fact -- the block GRANULARITY must be chosen for the whole
# line at once. Rather than committing to one fixed rule, we (a) build several candidate
# sequences at a spread of block granularities C, using two different construction
# schemes (round-robin cycling across all types at chunk size C, and majority-first
# chunking with single-job "breather" resets), (b) SELF-SCORE every candidate with the
# exact public cost model (the formula is fully public) and keep the true best, then
# (c) polish with adjacent-swap hill-climbing. This is a genuine reformulation --
# granularity search over the commitment structure -- not "greedy plus more iterations".
import sys, json


def compute_cost(pub, order, type_of):
    seq_types = [type_of[i] for i in order]
    N = len(order)
    total = 0.0
    for st in pub["stations"]:
        H = st["horizon"]; w = st["weight"]
        cold = st["cold_cost"]; warm = st["warm_cost"]
        fatigue_on = st["fatigue_on"]; R = st["fatigue_threshold"]; fcost = st["fatigue_cost"]
        window = []
        counts = {}
        run_len = 0
        prev_type = None
        station_cost = 0.0
        for i in range(N):
            t = seq_types[i]
            c = counts.get(t, 0)
            frac_cold = 1.0 - (min(c, H) / H)
            station_cost += warm[t] + (cold[t] - warm[t]) * frac_cold
            run_len = run_len + 1 if t == prev_type else 1
            if fatigue_on and run_len > R:
                station_cost += fcost[t] * (run_len - R)
            prev_type = t
            window.append(t)
            counts[t] = counts.get(t, 0) + 1
            if len(window) > H:
                old = window.pop(0)
                counts[old] -= 1
        total += w * station_cost
    return total


def by_type_groups(jobs):
    g = {}
    for j in jobs:
        g.setdefault(j["type"], []).append(j["id"])
    for t in g:
        g[t].sort()
    return g


def round_robin_seq(type_order, groups, C):
    remaining = {t: list(v) for t, v in groups.items()}
    seq = []
    while any(remaining.values()):
        progressed = False
        for t in type_order:
            if remaining[t]:
                seq.extend(remaining[t][:C])
                remaining[t] = remaining[t][C:]
                progressed = True
        if not progressed:
            break
    return seq


def majority_breather_seq(type_order, groups, C):
    remaining = {t: list(v) for t, v in groups.items()}
    seq = []
    while any(remaining.values()):
        avail = [t for t in type_order if remaining[t]]
        if not avail:
            break
        cur = avail[0]
        seq.extend(remaining[cur][:C])
        remaining[cur] = remaining[cur][C:]
        others = [t for t in type_order if t != cur and remaining[t]]
        if others and remaining[cur]:
            bt = others[0]
            seq.append(remaining[bt].pop(0))
    return seq


def solve(inst):
    n = inst["n"]
    jobs = inst["jobs"]
    type_of = {j["id"]: j["type"] for j in jobs}
    groups = by_type_groups(jobs)
    type_order = sorted(groups.keys(), key=lambda t: -len(groups[t]))
    n_types = len(groups)

    fatigue_thresholds = [st["fatigue_threshold"] for st in inst["stations"] if st["fatigue_on"]]
    R_min = min(fatigue_thresholds) if fatigue_thresholds else n
    H_max = max(st["horizon"] for st in inst["stations"])

    cand_C = sorted(set([1, 2, max(1, R_min // 2), R_min, R_min + 1, R_min + 2,
                          2 * R_min, H_max, 2 * H_max,
                          max(1, n // max(1, n_types)), n]))

    best_seq, best_cost = None, float("inf")
    for C in cand_C:
        if C < 1:
            continue
        for scheme in (round_robin_seq, majority_breather_seq):
            seq = scheme(type_order, groups, C)
            c = compute_cost(inst, seq, type_of)
            if c < best_cost:
                best_cost, best_seq = c, seq

    order = best_seq
    cur_cost = best_cost
    for _sweep in range(3):
        improved = False
        for i in range(len(order) - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            c = compute_cost(inst, order, type_of)
            if c < cur_cost - 1e-9:
                cur_cost = c
                improved = True
            else:
                order[i], order[i + 1] = order[i + 1], order[i]
        if not improved:
            break
    return order


inst = json.load(sys.stdin)
print(json.dumps({"order": solve(inst)}))

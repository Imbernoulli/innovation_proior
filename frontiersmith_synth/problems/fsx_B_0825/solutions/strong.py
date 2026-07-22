# TIER: strong
"""Exact optimal-stopping schedule via bitmask DP.

Key insight: a row's sweep cost (patrons still seated) is a KNOWN, non-increasing
step function of time, and feasibility only cares about the CUMULATIVE number of
rows swept by each round (never which ones). So the problem is an assignment
problem -- choose which rows to sweep and WHEN -- solvable exactly by dynamic
programming over (round, set-of-swept-rows), pruned against the free-row-floor
deadlines. This lets the plan proactively sweep an unpromising-but-stable row
early to bank slack, so it can defer any row that is about to shed most of its
patrons until just after that happens.
"""
import sys
from collections import Counter


def compute_req(T, F0, Fmin, demand_times):
    dc = Counter(demand_times)
    req = [0] * (T + 1)
    free = F0
    cum = 0
    for t in range(1, T + 1):
        free -= dc.get(t, 0)
        while free < Fmin:
            free += 1
            cum += 1
        req[t] = cum
    return req


def forced_rounds_from_req(req, T):
    return [t for t in range(1, T + 1) if req[t] > req[t - 1]]


def live_count(deaths, t):
    return sum(1 for d in deaths if d == -1 or d > t)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it)); F0 = int(next(it)); Fmin = int(next(it))
    segs = []
    for _ in range(N):
        k = int(next(it))
        segs.append([int(next(it)) for _ in range(k)])
    D = int(next(it))
    demand_times = [int(next(it)) for _ in range(D)]

    req = compute_req(T, F0, Fmin, demand_times)
    checkpoints = set(forced_rounds_from_req(req, T))

    cost_table = [[live_count(d, t) for t in range(T + 1)] for d in segs]

    FULL = 1 << N
    INF = float("inf")
    dp = [INF] * FULL
    dp[0] = 0
    # src_hist[t][mask] = (prev_mask, g)  with g=-1 meaning "no sweep this round"
    src_hist = [None] * (T + 1)

    for t in range(1, T + 1):
        ndp = dp[:]
        nsrc = [(m, -1) for m in range(FULL)]
        for mask in range(FULL):
            base = dp[mask]
            if base == INF:
                continue
            row = cost_table
            for g in range(N):
                bit = 1 << g
                if mask & bit:
                    continue
                nmask = mask | bit
                cand = base + row[g][t]
                if cand < ndp[nmask]:
                    ndp[nmask] = cand
                    nsrc[nmask] = (mask, g)
        if t in checkpoints:
            need = req[t]
            for mask in range(FULL):
                if ndp[mask] != INF and bin(mask).count("1") < need:
                    ndp[mask] = INF
        dp = ndp
        src_hist[t] = nsrc

    need_final = req[T]
    best_mask, best_cost = None, INF
    for mask in range(FULL):
        if dp[mask] < best_cost and bin(mask).count("1") >= need_final:
            best_cost, best_mask = dp[mask], mask

    plan = []
    mask = best_mask
    for t in range(T, 0, -1):
        prev_mask, g = src_hist[t][mask]
        if g != -1:
            plan.append((t, g + 1))
        mask = prev_mask
    plan.sort()

    out = [str(len(plan))]
    for (t, g) in plan:
        out.append(f"{t} {g}")
    print("\n".join(out))


if __name__ == "__main__":
    main()

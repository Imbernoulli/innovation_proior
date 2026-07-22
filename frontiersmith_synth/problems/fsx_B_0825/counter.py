#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic checker for the night-custodian
row-sweep schedule. Validates feasibility strictly, counts copied ("escorted")
seats, and scores against the checker's own naive index-order baseline.
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


def parse_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); T = int(next(it)); F0 = int(next(it)); Fmin = int(next(it))
    segs = []
    for _ in range(N):
        k = int(next(it))
        deaths = [int(next(it)) for _ in range(k)]
        segs.append(deaths)
    D = int(next(it))
    demand_times = [int(next(it)) for _ in range(D)]
    return N, T, F0, Fmin, segs, demand_times


def baseline_cost(N, T, segs, req, fr):
    cleaned = [False] * N
    cost = 0
    nxt = 0
    for r in fr:
        while nxt < N and cleaned[nxt]:
            nxt += 1
        if nxt >= N:
            break
        cleaned[nxt] = True
        cost += live_count(segs[nxt], r)
        nxt += 1
    return cost


def reject(msg):
    print("INFEASIBLE:", msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, T, F0, Fmin, segs, demand_times = parse_input(in_path)
    req = compute_req(T, F0, Fmin, demand_times)
    fr = forced_rounds_from_req(req, T)
    B = baseline_cost(N, T, segs, req, fr)

    try:
        with open(out_path) as f:
            toks = f.read().split()
        it = iter(toks)
        K_str = next(it)
        K = int(K_str)
        if K < 0 or K > N:
            reject(f"K={K} out of range [0,{N}]")
        plan = []
        for _ in range(K):
            t = int(next(it))
            g = int(next(it))
            plan.append((t, g))
        leftover = list(it)
        if leftover:
            reject(f"trailing tokens: {len(leftover)}")
    except StopIteration:
        reject("missing tokens / truncated output")
        return
    except ValueError:
        reject("non-integer token (or nan/inf) in output")
        return

    ts_seen = set()
    gs_seen = set()
    clean_at = {}
    for (t, g) in plan:
        if not (1 <= t <= T):
            reject(f"round {t} out of range [1,{T}]")
        if not (1 <= g <= N):
            reject(f"row {g} out of range [1,{N}]")
        if t in ts_seen:
            reject(f"round {t} used twice (custodian sweeps at most one row per round)")
        if g in gs_seen:
            reject(f"row {g} swept twice")
        ts_seen.add(t)
        gs_seen.add(g)
        clean_at[t] = g

    demand_count = Counter(demand_times)
    free = F0
    cost = 0
    if free < Fmin:
        reject("initial free-row stock already below the floor")
    for t in range(1, T + 1):
        if t in clean_at:
            g = clean_at[t]
            cost += live_count(segs[g - 1], t)
            free += 1
        free -= demand_count.get(t, 0)
        if free < Fmin:
            reject(f"free-row stock {free} < floor {Fmin} at round {t}")

    F = cost
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"B={B} F={F} K={K}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

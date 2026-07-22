# TIER: greedy
"""Reactive space-yield sweeper: only sweep when the free-row stock is about to
breach the floor, and when forced, sweep whichever row currently holds the fewest
still-seated patrons. No lookahead at future checkouts."""
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
    fr = forced_rounds_from_req(req, T)

    cleaned = [False] * N
    plan = []
    for r in fr:
        best_g, best_cost = None, None
        for g in range(N):
            if cleaned[g]:
                continue
            lc = live_count(segs[g], r)
            if best_cost is None or lc < best_cost:
                best_cost, best_g = lc, g
        if best_g is None:
            break
        cleaned[best_g] = True
        plan.append((r, best_g + 1))

    out = [str(len(plan))]
    for (t, g) in plan:
        out.append(f"{t} {g}")
    print("\n".join(out))


if __name__ == "__main__":
    main()

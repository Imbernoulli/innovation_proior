# TIER: trivial
"""Naive fixed-order sweep: whenever a row must be freed, sweep the lowest-numbered
row not yet swept -- ignore how many patrons are still seated in it."""
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
    nxt = 0
    for r in fr:
        while nxt < N and cleaned[nxt]:
            nxt += 1
        if nxt >= N:
            break
        cleaned[nxt] = True
        plan.append((r, nxt + 1))
        nxt += 1

    out = [str(len(plan))]
    for (t, g) in plan:
        out.append(f"{t} {g}")
    print("\n".join(out))


if __name__ == "__main__":
    main()

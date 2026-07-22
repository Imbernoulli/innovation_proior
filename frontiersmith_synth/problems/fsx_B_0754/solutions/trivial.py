# TIER: trivial
"""Replicates the checker's own reference construction: spend the budget in DOLLARS
proportional to each link's EXISTING capacity ("top up the biggest trunk first"). A
plausible-sounding rule that ignores flow-count and scarcity entirely."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    E = int(next(it)); F = int(next(it)); budget = float(next(it))
    cap = {}; cost = {}
    for e in range(E):
        cap[e] = float(next(it)); cost[e] = float(next(it))
    flows = []
    for _ in range(F):
        L = int(next(it))
        route = [int(next(it)) for _ in range(L)]
        flows.append(route)

    fc = {e: 0 for e in range(E)}
    for route in flows:
        for e in route:
            fc[e] += 1
    active = [e for e in range(E) if fc[e] > 0]
    totcap = sum(cap[e] for e in active) if active else 1.0

    x = [0.0] * E
    for e in active:
        dollars = budget * (cap[e] / totcap)
        x[e] = dollars / cost[e]

    print(" ".join("%.9f" % v for v in x))


if __name__ == "__main__":
    main()

# TIER: greedy
"""The obvious first idea: rank links by how many flows cross them per dollar of cost,
and dump the ENTIRE budget on the single top-ranked link -- "fix the busiest link /
global min-cut". Ignores that a link's TRUE marginal value depends on whether the flows
crossing it are already bottlenecked elsewhere (the nonlinear shadow-price coupling)."""
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

    scores = {e: fc[e] / cost[e] for e in range(E) if fc[e] > 0}
    x = [0.0] * E
    if scores:
        best = max(scores, key=lambda e: (scores[e], -e))
        x[best] = budget / cost[best]

    print(" ".join("%.9f" % v for v in x))


if __name__ == "__main__":
    main()

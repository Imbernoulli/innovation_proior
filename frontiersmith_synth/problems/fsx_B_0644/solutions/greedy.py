# TIER: greedy
"""The 'obvious' first attempt: treat this as a fractional knapsack over EDGES
in isolation.  For every edge, each successive upgrade level removes a known
chunk of that edge's OWN failure probability (p0 * 0.5**(level-1) * 0.5); rank
every (edge, level) item by isolated value-per-cost and buy the best-scoring
affordable items first, in order, never looking at the graph at all.

This ignores that an edge's TRUE marginal value depends on how much
redundancy already backs it up (a lone bridge edge vs. one of several
parallel edges in a fan) -- it just chases the biggest per-dollar probability
delta of each edge taken alone.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = int(toks[idx]); idx += 1
    idx += k
    edges = []
    for _ in range(m):
        u, v, p0, c, mu = (int(toks[idx + j]) for j in range(5))
        idx += 5
        edges.append((u, v, p0, c, mu))

    items = []  # (score, edge_idx, level, cost)
    for i, (u, v, p0, c, mu) in enumerate(edges):
        p0f = p0 / 1000.0
        cur = p0f
        for lvl in range(1, mu + 1):
            dp = cur * 0.5  # failure-prob removed by buying this level
            items.append((dp / c, i, lvl, c))
            cur *= 0.5
    items.sort(key=lambda x: -x[0])

    levels = [0] * m
    budget = B
    for score, i, lvl, c in items:
        if lvl == levels[i] + 1 and c <= budget:
            levels[i] = lvl
            budget -= c

    print("\n".join(str(x) for x in levels))


if __name__ == "__main__":
    main()

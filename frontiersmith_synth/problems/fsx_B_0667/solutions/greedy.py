# TIER: greedy
"""Textbook one-shot spectral fill: compute the Fiedler vector of the base
(all-weight-1) network ONCE, rank every cable by that static sensitivity
score (x_u - x_v)^2, and fill cables in that fixed rank order -- pushing
each one to its own cap/budget limit before moving to the next.  This is
the obvious "reinforce whatever the spectrum says matters most" recipe.
It never re-examines the ranking as weights change, so it cannot notice
that a cable it just maxed out has stopped being useful (diminishing
marginal return / node-cap saturation), nor that splitting the budget
across several parallel cross-cluster cables beats dumping it all on the
first one the static ranking likes."""
import sys

import numpy as np


def read_instance():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    w_budget = int(next(it))
    edges = [(int(next(it)), int(next(it))) for _ in range(m)]
    cap = [int(next(it)) for _ in range(n)]
    return n, m, w_budget, edges, cap


def fiedler_vector(n, edges, w):
    lap = np.zeros((n, n))
    for (u, v), wt in zip(edges, w):
        lap[u, u] += wt
        lap[v, v] += wt
        lap[u, v] -= wt
        lap[v, u] -= wt
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)
    return vecs[:, order[1]]


def main():
    n, m, w_budget, edges, cap = read_instance()
    deg0 = [0] * n
    for (u, v) in edges:
        deg0[u] += 1
        deg0[v] += 1

    x = fiedler_vector(n, edges, [1] * m)
    sens = [(x[u] - x[v]) ** 2 for (u, v) in edges]
    rank = sorted(range(m), key=lambda e: (-sens[e], e))

    remaining = [cap[i] - deg0[i] for i in range(n)]
    w = [1] * m
    budget_left = w_budget
    for e in rank:
        if budget_left <= 0:
            break
        u, v = edges[e]
        add = min(budget_left, remaining[u], remaining[v])
        if add > 0:
            w[e] += add
            remaining[u] -= add
            remaining[v] -= add
            budget_left -= add

    sys.stdout.write("\n".join(str(x) for x in w) + "\n")


if __name__ == "__main__":
    main()

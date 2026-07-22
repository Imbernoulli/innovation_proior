# TIER: strong
"""Electrical-network insight: the cap that blocks the bottleneck cable is
a node-local constraint, but algebraic connectivity is a GLOBAL, concave
function of the weight vector (lambda_2(w) = min_{x perp 1,||x||=1} sum_e
w_e (x_u-x_v)^2, a minimum of linear functions of w).  So instead of
trusting one static ranking, we run cap-aware coordinate ascent: spend the
budget in small increments, always on the edge with the CURRENT highest
Fiedler-sensitivity among edges with spare node-cap room, and recompute
the Fiedler vector after every increment.  When the direct bottleneck
cable saturates its (tiny) node caps, the recomputed sensitivity
naturally reveals which detour cables -- cap-free parallel routes between
the two clusters -- are now the best marginal spend, so the budget spills
onto them instead of sitting idle or piling onto one already-saturated
pick."""
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


def algebraic_connectivity_and_vec(n, edges, w):
    lap = np.zeros((n, n))
    for (u, v), wt in zip(edges, w):
        lap[u, u] += wt
        lap[v, v] += wt
        lap[u, v] -= wt
        lap[v, u] -= wt
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)
    return float(vals[order[1]]), vecs[:, order[1]]


def main():
    n, m, w_budget, edges, cap = read_instance()
    deg0 = [0] * n
    for (u, v) in edges:
        deg0[u] += 1
        deg0[v] += 1

    remaining = [cap[i] - deg0[i] for i in range(n)]
    w = [1] * m
    budget_left = w_budget

    # chunk size keeps the eigh recomputation count bounded on large tests
    chunk = max(1, w_budget // 40)

    stall_guard = 0
    while budget_left > 0 and stall_guard < 4 * m + 200:
        stall_guard += 1
        _, x = algebraic_connectivity_and_vec(n, edges, w)
        best_e, best_s = -1, -1.0
        for e, (u, v) in enumerate(edges):
            if remaining[u] <= 0 or remaining[v] <= 0:
                continue
            s = (x[u] - x[v]) ** 2
            if s > best_s:
                best_s, best_e = s, e
        if best_e < 0:
            break
        u, v = edges[best_e]
        add = min(chunk, budget_left, remaining[u], remaining[v])
        add = max(add, 1)
        w[best_e] += add
        remaining[u] -= add
        remaining[v] -= add
        budget_left -= add

    sys.stdout.write("\n".join(str(x) for x in w) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gen.py <testId> -- emits one instance of "Sprawl Doppelganger" to stdout.

Deterministic: all randomness is seeded from testId only.

Builds a small unicyclic "zoning template" H (a tree plus exactly one extra
edge, so H has a single cycle).  Prints H, the exact closed-walk moments
mu_1..mu_k of H (as integer numerators over the fixed denominator nH), the
degree-square-sum invariant, and calibrated tolerances.  The tolerances are
derived from two REFERENCE constructions built internally (a cyclic covering
lift, and a naive "random-tree-of-copies" expander) so that both a smart
covering-lift strategy and an obvious greedy-wiring strategy are guaranteed
feasible, while structure-blind constructions (bare big cycle, a padded
pendant tail) are not.
"""
import sys
from collections import defaultdict, deque
from fractions import Fraction


def bfs_dist(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for x in adj[u]:
            if dist[x] == -1:
                dist[x] = dist[u] + 1
                q.append(x)
    return dist


def diam(n, adj):
    d = 0
    for v in range(n):
        dist = bfs_dist(n, adj, v)
        d = max(d, max(dist))
    return d


def closed_walks_all(n, adj, k):
    """tw[j] = trace(A^j) = total number of closed walks of length j (all j=1..k)."""
    tw = [0] * (k + 1)
    for v in range(n):
        cnt = {v: 1}
        for t in range(1, k + 1):
            nxt = defaultdict(int)
            for u, w in cnt.items():
                for x in adj[u]:
                    nxt[x] += w
            cnt = nxt
            tw[t] += cnt.get(v, 0)
    return tw


def deg_sq_sum(n, adj):
    return sum(len(adj[v]) ** 2 for v in range(n))


def gen_H(nH, seed, min_leaves=3, min_maxdeg=4, tries=300):
    """Random unicyclic (tree + 1 edge) template with forced branching."""
    edges = adj = a = b = leaves = None
    for attempt in range(tries):
        rng = __import__("random").Random(seed * 7919 + 13 + attempt * 97)
        edges = []
        adj = defaultdict(list)
        for v in range(1, nH):
            u = 0 if rng.random() < 0.5 else rng.randrange(0, v)
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
        tree_edge_set = set(frozenset(e) for e in edges)
        cand = [(u, v) for u in range(nH) for v in range(u + 1, nH)
                if frozenset((u, v)) not in tree_edge_set]
        a, b = rng.choice(cand)
        edges.append((a, b))
        adj[a].append(b)
        adj[b].append(a)
        degs = [len(adj[v]) for v in range(nH)]
        leaves = [v for v in range(nH) if degs[v] == 1]
        if len(leaves) >= min_leaves and max(degs) >= min_maxdeg:
            return edges, adj, (a, b), leaves
    return edges, adj, (a, b), leaves  # fallback after exhausting tries


def build_lift(nH, edges, estar, L):
    """L-fold cyclic covering lift of H along its unique cycle (breaks estar,
    replicates the spanning tree L times, re-glues copies into one necklace)."""
    a, b = estar
    tree_edges = [e for e in edges if frozenset(e) != frozenset(estar)]
    N = nH * L
    adj = defaultdict(list)

    def idx(i, v):
        return i * nH + v

    for i in range(L):
        for (u, v) in tree_edges:
            adj[idx(i, u)].append(idx(i, v))
            adj[idx(i, v)].append(idx(i, u))
    for i in range(L):
        j = (i + 1) % L
        adj[idx(i, b)].append(idx(j, a))
        adj[idx(j, a)].append(idx(i, b))
    return N, adj


def build_naive_tree(nH, edges, C, leaf):
    """C copies of H wired into a balanced binary tree via one designated
    leaf vertex per copy -- the "obvious" way to spend a vertex budget and
    stay connected without any covering-space reasoning."""
    N = nH * C
    adj = defaultdict(list)

    def idx(i, v):
        return i * nH + v

    for i in range(C):
        for (u, v) in edges:
            adj[idx(i, u)].append(idx(i, v))
            adj[idx(i, v)].append(idx(i, u))
    for i in range(1, C):
        p = (i - 1) // 2
        adj[idx(i, leaf)].append(idx(p, leaf))
        adj[idx(p, leaf)].append(idx(i, leaf))
    return N, adj


def main():
    testId = int(sys.argv[1])
    k = 6
    nH = 7 + (testId - 1) % 6

    edges, adjH, estar, leaves = gen_H(nH, testId)
    leaf = leaves[0]
    dH = diam(nH, adjH)

    # girth of H = tree-path(a,b) + 1
    tree_edges = [e for e in edges if frozenset(e) != frozenset(estar)]
    adjt = defaultdict(list)
    for u, v in tree_edges:
        adjt[u].append(v)
        adjt[v].append(u)
    dist = {estar[0]: 0}
    q = deque([estar[0]])
    while q:
        u = q.popleft()
        for x in adjt[u]:
            if x not in dist:
                dist[x] = dist[u] + 1
                q.append(x)
    gH = dist[estar[1]] + 1

    L_target = max(6, min(70, round(13 * dH / gH)))
    n = nH * L_target

    twH = closed_walks_all(nH, adjH, k)
    s2H = deg_sq_sum(nH, adjH)

    # two internal reference constructions to calibrate tolerances
    N1, adj1 = build_lift(nH, edges, estar, L_target)
    tw1 = closed_walks_all(N1, adj1, k)
    s2_1 = deg_sq_sum(N1, adj1)
    dev1 = max(abs(Fraction(tw1[j], N1) - Fraction(twH[j], nH)) for j in range(1, k + 1))
    s2dev1 = abs(Fraction(s2_1, N1) - Fraction(s2H, nH))

    N2, adj2 = build_naive_tree(nH, edges, L_target, leaf)
    tw2 = closed_walks_all(N2, adj2, k)
    s2_2 = deg_sq_sum(N2, adj2)
    dev2 = max(abs(Fraction(tw2[j], N2) - Fraction(twH[j], nH)) for j in range(1, k + 1))
    s2dev2 = abs(Fraction(s2_2, N2) - Fraction(s2H, nH))

    eps = Fraction(14, 10) * max(dev1, dev2) + Fraction(1, 4 * nH)
    eps2 = Fraction(14, 10) * max(s2dev1, s2dev2) + Fraction(1, 4 * nH)

    maxdegH = max(len(adjH[v]) for v in range(nH))
    D_MAX = min(12, maxdegH + 5)
    M_MAX = 4 * n

    out = []
    out.append(f"{nH} {k} {n}")
    out.append(f"{eps.numerator} {eps.denominator}")
    out.append(f"{eps2.numerator} {eps2.denominator}")
    out.append(f"{D_MAX} {M_MAX}")
    out.append(f"{len(edges)}")
    for (u, v) in edges:
        out.append(f"{u} {v}")
    for j in range(1, k + 1):
        out.append(f"{twH[j]}")
    out.append(f"{s2H}")
    print("\n".join(out))


if __name__ == "__main__":
    main()

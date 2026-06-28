#!/usr/bin/env python3
"""Deterministic local scorer for "Graph Coloring with Soft Conflicts".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score (higher is better).

Scoring rule (see context.md "Evaluation settings"):
  * The instance is a weighted undirected graph with n vertices, m edges, and a
    budget of k colors. A SOLUTION assigns each vertex a color in {0,...,k-1};
    it is a list of n integers (color[i] for i = 0..n-1).
  * An edge (u,v,w) is a CONFLICT if color[u] == color[v]; its cost is w.
    The cost of a coloring is L = sum of w over all conflicting edges
    (the total weight of monochromatic edges) -- we MINIMIZE L.
  * FEASIBILITY: the output must be exactly n integers, each in [0, k-1]. If it
    is not (wrong count, an out-of-range color, garbage tokens, missing file),
    the solution is INFEASIBLE and the score is 0 (the feasibility floor).
  * To turn "minimize L" into a higher-is-better continuous score, the scorer
    recomputes a deterministic GREEDY baseline conflict weight G (a DSATUR-style
    greedy that it computes itself, so the reference is reproducible and
    independent of the solver) and reports

        score = round(1_000_000 * (G + 1) / (L + 1))     (feasible)
        score = 0                                        (infeasible)

    The +1 smoothing keeps the score finite and well-defined even when a
    coloring achieves L = 0 (zero conflicts). The greedy baseline scores
    exactly 1_000_000 against itself; a lower-conflict coloring scores strictly
    more; a worse one scores less but stays positive.

The scorer is self-contained and deterministic: it does not trust the solver
and recomputes G itself.
"""
import re
import sys


INT_TOKEN_RE = re.compile(r"^[+-]?[0-9]+$")


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    k = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        edges.append((u, v, w))
    return n, m, k, edges


def read_solution(path, n, k):
    """Return a list of n ints each in [0,k-1], else None (infeasible)."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != n:
        return None
    col = []
    for t in toks:
        if not INT_TOKEN_RE.fullmatch(t):
            return None
        try:
            c = int(t)
        except ValueError:
            return None
        if c < 0 or c >= k:
            return None
        col.append(c)
    return col


def conflict_weight(col, edges):
    L = 0
    for (u, v, w) in edges:
        if col[u] == col[v]:
            L += w
    return L


def greedy_baseline(n, k, edges):
    """Deterministic DSATUR-style greedy baseline.

    Build weighted adjacency, order vertices by descending weighted degree
    (ties broken by smaller index), and assign each the color minimizing the
    weight of conflicts with already-colored neighbours (ties -> smaller color).
    This is a reasonable, fully reproducible reference -- not the optimum.
    """
    adj = [[] for _ in range(n)]   # adj[u] = list of (v, w)
    wdeg = [0] * n
    for (u, v, w) in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
        wdeg[u] += w
        wdeg[v] += w
    order = sorted(range(n), key=lambda i: (-wdeg[i], i))
    color = [-1] * n
    for u in order:
        # cost of giving u each color = weight of already-colored neighbours
        # that share that color.
        cost = [0] * k
        for (v, w) in adj[u]:
            cv = color[v]
            if cv >= 0:
                cost[cv] += w
        best_c = 0
        best_cost = cost[0]
        for c in range(1, k):
            if cost[c] < best_cost:
                best_cost = cost[c]
                best_c = c
        color[u] = best_c
    return conflict_weight(color, edges)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, k, edges = read_instance(sys.argv[1])

    col = read_solution(sys.argv[2], n, k)
    if col is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    L = conflict_weight(col, edges)
    G = greedy_baseline(n, k, edges)
    score = int(round(1_000_000.0 * (G + 1.0) / (L + 1.0)))
    print(score)


if __name__ == "__main__":
    main()

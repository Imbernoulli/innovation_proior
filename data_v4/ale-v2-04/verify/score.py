#!/usr/bin/env python3
"""Deterministic local scorer for "Graph Coloring (Minimize Colors)".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score (higher is better).

Scoring rule (see context.md "Evaluation settings"):
  * The instance is a simple undirected graph with n vertices and m edges.
    A SOLUTION is a PROPER coloring: a list of n integers color[i] (i = 0..n-1),
    color[i] >= 0, such that for every edge (u, v) we have color[u] != color[v].
  * The objective is to MINIMIZE the number of DISTINCT colors actually used,
    K = |{ color[i] : 0 <= i < n }|. Fewer colors is better.
  * FEASIBILITY (the floor): the output must be exactly n integers, each a
    non-negative integer, AND the coloring must be PROPER (no edge has both
    endpoints the same color). If any of that fails -- wrong count, a negative or
    non-integer token, a missing file, or ANY monochromatic edge -- the solution
    is INFEASIBLE and the score is 0.

    Note: only the NUMBER of distinct colors matters, not their labels. Colors do
    not need to be a contiguous range {0..K-1}; using labels {0, 5, 9} counts as
    K = 3 colors. (Any proper coloring can be relabeled to {0..K-1} without
    changing K, so this is purely a convenience.)

  * To turn "minimize K" into a higher-is-better continuous score, the scorer
    recomputes a deterministic GREEDY baseline color count G (a first-fit greedy
    over vertices in descending-degree order, which it computes itself, so the
    reference is reproducible and independent of the solver) and reports

        score = round(1_000_000 * G / K)     (feasible, K >= 1)
        score = 0                            (infeasible)

    Using FEWER colors than greedy (K < G) scores strictly above 1_000_000;
    matching greedy scores exactly 1_000_000; using more scores below it but stays
    positive. K = 0 can only happen when n = 0, handled below.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes G itself. The feasibility check (properness) is what enforces the
ALE-Bench "invalid output -> 0" floor.
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
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        edges.append((u, v))
    return n, m, edges


def read_solution(path, n):
    """Return a list of n non-negative ints, else None (malformed)."""
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
        if c < 0:
            return None
        col.append(c)
    return col


def is_proper(col, edges):
    for (u, v) in edges:
        if col[u] == col[v]:
            return False
    return True


def num_colors(col):
    return len(set(col))


def greedy_baseline(n, edges):
    """Deterministic first-fit greedy baseline color count.

    Build adjacency, order vertices by descending degree (ties broken by smaller
    index), and assign each vertex the smallest color not used by an
    already-colored neighbour. Always proper; returns the number of distinct
    colors it uses. This is a reasonable, fully reproducible reference -- not the
    optimum (i.e. not the chromatic number).
    """
    adj = [[] for _ in range(n)]
    deg = [0] * n
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
    order = sorted(range(n), key=lambda i: (-deg[i], i))
    color = [-1] * n
    for u in order:
        used = set()
        for v in adj[u]:
            if color[v] >= 0:
                used.add(color[v])
        c = 0
        while c in used:
            c += 1
        color[u] = c
    return num_colors(color) if n > 0 else 0


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, edges = read_instance(sys.argv[1])

    col = read_solution(sys.argv[2], n)
    if col is None:
        print(0)  # malformed -> INFEASIBLE -> floored to 0
        return
    if not is_proper(col, edges):
        print(0)  # improper coloring -> INFEASIBLE -> floored to 0
        return

    if n == 0:
        # vacuous instance: no vertices, no colors needed.
        print(1_000_000)
        return

    K = num_colors(col)            # distinct colors actually used
    G = greedy_baseline(n, edges)  # deterministic greedy reference
    # K >= 1 here because n >= 1 and the coloring is proper.
    score = int(round(1_000_000.0 * G / K))
    print(score)


if __name__ == "__main__":
    main()

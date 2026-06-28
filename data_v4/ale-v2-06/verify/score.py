#!/usr/bin/env python3
"""Deterministic local scorer for "Dense Weighted Independent Set".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Problem recap (see context.md "Evaluation settings"):
  * The instance is an undirected graph on n vertices with positive integer vertex
    weights w_i and m edges. The objective is to choose a subset S of vertices that
    is an INDEPENDENT SET (no edge has both endpoints in S) maximizing sum_{i in S} w_i.
  * A SOLUTION is an integer k (0 <= k <= n) followed by k DISTINCT vertex ids in
    [0, n-1] -- the chosen set. Order is irrelevant. k = 0 is allowed (empty set,
    weight 0).

FEASIBILITY (floor to 0):
  The output must be a single integer k followed by exactly k integer tokens, all in
  [0, n-1], pairwise DISTINCT, the declared count k matching the number of ids, AND
  the set must be a valid INDEPENDENT SET (no two chosen vertices are adjacent). If
  any of this fails -- wrong count, an out-of-range id, a repeated id, the header k
  not matching the number of ids, garbage tokens, a missing file, OR an edge inside
  the chosen set -- the solution is INFEASIBLE and the score is 0. (k = 0 with no ids
  is feasible and has weight 0.)

SCORE (deterministic, reproducible, scale-invariant):
  Let
    * W      = the solution's total weight (sum of chosen vertex weights),
    * W_base = the total weight of the scorer's own deterministic GREEDY independent
               set (GWMIN-style: repeatedly take the remaining vertex of largest
               weight whose neighbours are all still available, then forbid its
               neighbours), recomputed here independent of the solver, and
    * W_ref  = max(W_base, 1) a positive normalizer.
  Then
        score = round( 1_000_000 * W / W_ref ),  clamped to >= 0.
  The greedy baseline scores exactly 1_000_000. Any independent set heavier than the
  greedy one scores strictly more; a lighter (but still independent) one scores less
  but stays >= 0. The empty set (W = 0) scores 0. INFEASIBLE -> 0.

The scorer does not trust the solver: it recomputes W_base itself and re-checks
independence against the instance's edge set.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    w = [int(next(it)) for _ in range(n)]
    adj = [set() for _ in range(n)]
    edges = []
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
        edges.append((a, b))
    return n, m, w, adj, edges


def read_solution(path, n):
    """Return a list of k distinct ids in [0,n-1] if the file is well-formed, else
    None. Header k must match the number of id tokens. (Independence is checked
    separately by the caller.)"""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) < 1:
        return None
    try:
        k = int(toks[0])
    except ValueError:
        return None
    if k < 0 or k > n:
        return None
    if len(toks) != 1 + k:
        return None  # header count must match exactly
    ids = []
    seen = [False] * n
    for t in toks[1:]:
        try:
            v = int(t)
        except ValueError:
            return None
        if v < 0 or v >= n or seen[v]:
            return None
        seen[v] = True
        ids.append(v)
    return ids


def is_independent(ids, adj):
    """True iff no two vertices in `ids` are adjacent."""
    sset = set(ids)
    for v in ids:
        # check against the smaller of (v's neighbours) intersected with the set
        av = adj[v]
        if len(av) <= len(sset):
            for u in av:
                if u in sset:
                    return False
        else:
            for u in sset:
                if u in av:
                    return False
    return True


def greedy_gwmin(n, w, adj):
    """Deterministic greedy independent set: repeatedly take the available vertex of
    largest weight (ties: smaller id), then forbid its neighbours. Returns total
    weight."""
    avail = [True] * n
    order = sorted(range(n), key=lambda i: (-w[i], i))
    total = 0
    for v in order:
        if not avail[v]:
            continue
        total += w[v]
        avail[v] = False
        for u in adj[v]:
            avail[u] = False
    return total


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, w, adj, edges = read_instance(sys.argv[1])

    ids = read_solution(sys.argv[2], n)
    if ids is None:
        print(0)  # INFEASIBLE (malformed) -> floored to 0
        return
    if not is_independent(ids, adj):
        print(0)  # INFEASIBLE (not an independent set) -> floored to 0
        return

    W = sum(w[v] for v in ids)
    W_base = greedy_gwmin(n, w, adj)
    W_ref = W_base if W_base > 0 else 1

    score = 1_000_000.0 * W / W_ref
    if score < 0.0:
        score = 0.0
    print(int(round(score)))


if __name__ == "__main__":
    main()

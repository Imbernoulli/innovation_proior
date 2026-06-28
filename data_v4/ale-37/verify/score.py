#!/usr/bin/env python3
"""Deterministic local scorer for "Quadratic Assignment Placement" (QAP).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is a Quadratic Assignment Problem of size n: an n x n integer
    FLOW matrix f[i][j] (flow between facilities i and j) and an n x n integer
    DISTANCE matrix d[k][l] (distance between locations k and l). A SOLUTION is a
    PERMUTATION p of {0,...,n-1}: p[i] is the LOCATION that facility i is placed
    on. The objective (lower is better) is the quadratic arrangement cost

        cost(p) = sum_{i=0..n-1} sum_{j=0..n-1} f[i][j] * d[p[i]][p[j]].

  * FEASIBILITY: the output must parse as exactly n integers that form a
    PERMUTATION of {0,...,n-1} (each location used exactly once, all in range).
    Anything else -- a parse error, wrong count, an out-of-range location, or a
    repeated location -- is INFEASIBLE and the score is 0 (the feasibility floor).

  * The reference is the IDENTITY permutation p[i] = i, whose cost cost_id is
    recomputed inside the scorer so the baseline is reproducible and
    solver-independent.

  * SCORE = round(1_000_000 * cost_id / cost_solver) for a feasible permutation
    with cost_solver > 0, and a generous full-credit cap when cost_solver == 0
    (a zero-cost placement, essentially never reachable for these instances).
    The identity reference scores ~1_000_000; a better (lower-cost) permutation
    scores strictly MORE; a worse-but-feasible one scores less but stays
    positive. Infeasible -> 0.

The scorer is self-contained and deterministic: it recomputes the identity
reference itself, so the baseline is reproducible and solver-independent.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    f = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            f[i][j] = int(next(it))
    d = [[0] * n for _ in range(n)]
    for k in range(n):
        for l in range(n):
            d[k][l] = int(next(it))
    return n, f, d


def read_solution(path, n):
    """Return a list p of n locations forming a permutation, or None on error."""
    try:
        with open(path) as fh:
            toks = fh.read().split()
    except OSError:
        return None
    if len(toks) != n:
        return None
    p = []
    seen = [False] * n
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None
        if v < 0 or v >= n:
            return None      # location out of range
        if seen[v]:
            return None      # repeated location -> not a permutation
        seen[v] = True
        p.append(v)
    # every location 0..n-1 used exactly once is guaranteed by the count + seen
    return p


def qap_cost(n, f, d, p):
    """cost(p) = sum_i sum_j f[i][j] * d[p[i]][p[j]]  (use Python big ints)."""
    total = 0
    for i in range(n):
        fi = f[i]
        dpi = d[p[i]]
        for j in range(n):
            fij = fi[j]
            if fij:
                total += fij * dpi[p[j]]
    return total


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, f, d = read_instance(sys.argv[1])

    # reference (identity) cost -- recomputed by the scorer, solver-independent
    ident = list(range(n))
    cost_id = qap_cost(n, f, d, ident)

    p = read_solution(sys.argv[2], n)
    if p is None:
        print(0)  # parse error / wrong count / not a permutation -> infeasible
        return

    cost = qap_cost(n, f, d, p)
    if cost <= 0:
        # a zero (or non-positive) cost placement: unreachable for these
        # non-negative flow/distance instances with positive total flow, but
        # give full+ credit rather than divide by zero.
        print(2_000_000)
        return

    # cost_id could in principle be 0 only if all flows or all distances vanish;
    # the generator guarantees a strictly positive identity cost, but guard it.
    if cost_id <= 0:
        print(1_000_000)
        return

    score = int(round(1_000_000.0 * cost_id / cost))
    print(score)


if __name__ == "__main__":
    main()

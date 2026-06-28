#!/usr/bin/env python3
"""Deterministic local scorer for "Facility Layout Assignment" (QAP).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance has n facilities/locations, an n x n flow matrix F and an
    n x n distance matrix D, all non-negative integers.
  * A SOLUTION is a permutation p[0..n-1] of {0,...,n-1}: facility i is placed
    on location p[i].
  * The cost of an assignment is the quadratic-assignment objective
        C(p) = sum_{i,j} F[i][j] * D[p[i]][p[j]]
    (lower is better).
  * Let C0 = C(identity), the cost of placing facility i on location i. This is
    the trivial baseline; it is recomputed inside the scorer so the reference is
    reproducible and independent of the solver.
  * FEASIBILITY: the output must be exactly n integers forming a permutation of
    {0,...,n-1}. If it is not (wrong count, out-of-range index, repeats, garbage
    tokens, missing file), the solution is INFEASIBLE and the score is 0.
  * SCORE = round(1_000_000 * C0 / C(p)) for a feasible p with C(p) > 0.
    The identity baseline scores exactly 1_000_000; a cheaper (better) assignment
    scores strictly more; a worse assignment scores less but stays positive.
    Infeasible -> 0. (A degenerate all-zero-cost instance gives full credit.)

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes C0 itself.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    F = [[0] * n for _ in range(n)]
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            F[i][j] = int(next(it))
    for i in range(n):
        for j in range(n):
            D[i][j] = int(next(it))
    return n, F, D


def read_solution(path, n):
    """Return a list of n ints if the file is a clean permutation, else None."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != n:
        return None
    perm = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None
        perm.append(v)
    seen = [False] * n
    for v in perm:
        if v < 0 or v >= n or seen[v]:
            return None
        seen[v] = True
    return perm


def qap_cost(perm, F, D):
    """C(perm) = sum_{i,j} F[i][j] * D[perm[i]][perm[j]]."""
    n = len(perm)
    total = 0
    for i in range(n):
        pi = perm[i]
        Fi = F[i]
        Dpi = D[pi]
        for j in range(n):
            f = Fi[j]
            if f:
                total += f * Dpi[perm[j]]
    return total


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, F, D = read_instance(sys.argv[1])

    perm = read_solution(sys.argv[2], n)
    if perm is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    identity = list(range(n))
    C0 = qap_cost(identity, F, D)
    Cp = qap_cost(perm, F, D)

    if C0 <= 0:
        # Degenerate instance with zero baseline cost: any feasible perm is
        # optimal; award full credit.
        print(1_000_000)
        return
    if Cp <= 0:
        # Feasible perm achieving zero cost: better than any positive cost.
        # Cap at a large but finite full-credit-plus value to stay an integer.
        print(1_000_000_000)
        return

    score = int(round(1_000_000.0 * C0 / Cp))
    print(score)


if __name__ == "__main__":
    main()

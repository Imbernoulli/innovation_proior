#!/usr/bin/env python3
"""Deterministic local scorer for "Drone Survey Sweep".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance has n stations with integer coordinates.
  * A SOLUTION is a permutation p[0..n-1] of {0,...,n-1} describing the visiting
    order of a CLOSED tour (after p[n-1] the drone returns to p[0]). Visiting every
    station exactly once means every station has degree exactly 2 in the closed
    tour, i.e. it is a degree-<=2 spanning structure that spans all stations.
  * Let L = sum of Euclidean edge lengths around that closed tour.
  * Let G = the closed-tour length of the deterministic greedy nearest-neighbour
    baseline started at station 0 (the same construction the scorer recomputes
    here, so the baseline is reproducible and independent of the solver).
  * FEASIBILITY: the output must be exactly n integers forming a permutation of
    {0,...,n-1}. If it is not (wrong count, out-of-range index, repeats, garbage
    tokens, missing file), the solution is INFEASIBLE and the score is 0.
  * SCORE = round(1_000_000 * G / L) for a feasible tour with L > 0.
    The greedy baseline scores exactly 1_000_000; a shorter (better) tour scores
    strictly more; a longer tour scores less but stays positive. Infeasible -> 0.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes G itself.
"""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    xs = [0] * n
    ys = [0] * n
    for i in range(n):
        xs[i] = int(next(it))
        ys[i] = int(next(it))
    return n, xs, ys


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


def tour_length(perm, xs, ys):
    n = len(perm)
    total = 0.0
    for i in range(n):
        a = perm[i]
        b = perm[(i + 1) % n]
        dx = xs[a] - xs[b]
        dy = ys[a] - ys[b]
        total += math.hypot(dx, dy)
    return total


def greedy_nn_length(n, xs, ys):
    """Deterministic nearest-neighbour closed-tour length from station 0."""
    if n <= 1:
        return 0.0
    visited = [False] * n
    order = [0]
    visited[0] = True
    cur = 0
    for _ in range(n - 1):
        best = -1
        best_d = None
        cx, cy = xs[cur], ys[cur]
        for j in range(n):
            if visited[j]:
                continue
            dx = cx - xs[j]
            dy = cy - ys[j]
            d = dx * dx + dy * dy  # compare on squared distance (monotone, exact)
            if best_d is None or d < best_d or (d == best_d and j < best):
                best_d = d
                best = j
        visited[best] = True
        order.append(best)
        cur = best
    return tour_length(order, xs, ys)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, xs, ys = read_instance(sys.argv[1])

    perm = read_solution(sys.argv[2], n)
    if perm is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    if n <= 1:
        # A single (or zero) station: any valid permutation is optimal; full credit.
        print(1_000_000)
        return

    L = tour_length(perm, xs, ys)
    if L <= 0.0:
        # All coordinates distinct by construction, so a real tour has L > 0.
        # Guard against degeneracy: treat zero-length as infeasible.
        print(0)
        return

    G = greedy_nn_length(n, xs, ys)
    score = int(round(1_000_000.0 * G / L))
    print(score)


if __name__ == "__main__":
    main()

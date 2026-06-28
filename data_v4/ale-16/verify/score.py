#!/usr/bin/env python3
"""Deterministic local scorer for "Prize-Collecting Patrol".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Problem recap (see context.md "Evaluation settings"):
  * The instance has a DEPOT (always visited, no prize) and n optional nodes, each
    with integer coordinates and an integer prize > 0.
  * A SOLUTION is an integer k (0 <= k <= n) followed by k distinct node ids in
    [0, n-1], the visiting ORDER of a CLOSED tour
        depot -> p[0] -> p[1] -> ... -> p[k-1] -> depot.
    k = 0 is allowed and denotes "visit nothing" (the tour is the empty loop at the
    depot, profit 0).
  * PROFIT of a feasible solution:
        profit = (sum of prizes of the visited nodes) - (total Euclidean travel),
    where total Euclidean travel is the length of the closed tour above (0 when
    k = 0). Profit may be positive, zero, or negative -- it is NOT the score.

FEASIBILITY (floor to 0):
  The output must be a single integer k followed by exactly k integer tokens, all
  in [0, n-1], pairwise distinct, and the declared count must match. If any of this
  fails -- wrong count, an out-of-range id, a repeated id, the header k not matching
  the number of ids, garbage tokens, a missing file -- the solution is INFEASIBLE
  and the score is 0. (k = 0 with no ids is feasible.)

SCORE (deterministic, reproducible, scale-invariant):
  Let
    * P      = the solution's profit,
    * P_base = the profit of the scorer's own deterministic "visit-all" tour built
               by nearest-neighbour from the depot over ALL n nodes (recomputed
               here, independent of the solver), and
    * D      = the closed-tour Euclidean LENGTH of that same visit-all NN tour
               (a positive instance-scale normalizer; D > 0 for n >= 1).
  Then
        score = round( 1_000_000 + 1_000_000 * (P - P_base) / D ),  clamped to >= 0.
  The visit-all NN baseline scores exactly 1_000_000. Any solution with higher
  profit scores strictly more; a worse one scores less but stays >= 0. The empty
  tour (P = 0) scores 1_000_000 + round(1_000_000 * (-P_base)/D), which is > 1e6
  whenever visiting everything is a net loss (the typical case here) -- so a real
  solver must beat *that* too, not merely the visit-all tour. INFEASIBLE -> 0.

The scorer does not trust the solver: it recomputes P_base and D itself.
"""
import sys
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    dx = int(next(it))
    dy = int(next(it))
    xs = [0] * n
    ys = [0] * n
    pr = [0] * n
    for i in range(n):
        xs[i] = int(next(it))
        ys[i] = int(next(it))
        pr[i] = int(next(it))
    return n, dx, dy, xs, ys, pr


def read_solution(path, n):
    """Return a list of k distinct ids in [0,n-1] if the file is well-formed,
    else None. Header k must match the number of id tokens."""
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


def euclid(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def tour_profit(ids, dx, dy, xs, ys, pr):
    """Profit = sum prizes - closed-tour length (depot -> ids... -> depot)."""
    if not ids:
        return 0.0
    prize = sum(pr[v] for v in ids)
    dist = 0.0
    px, py = dx, dy
    for v in ids:
        dist += euclid(px, py, xs[v], ys[v])
        px, py = xs[v], ys[v]
    dist += euclid(px, py, dx, dy)  # return to depot
    return prize - dist


def visit_all_nn(n, dx, dy, xs, ys, pr):
    """Deterministic nearest-neighbour tour over ALL nodes from the depot.
    Returns (profit, closed_tour_length)."""
    if n <= 0:
        return 0.0, 0.0
    visited = [False] * n
    order = []
    cx, cy = float(dx), float(dy)
    for _ in range(n):
        best = -1
        best_d = None
        for j in range(n):
            if visited[j]:
                continue
            ddx = cx - xs[j]
            ddy = cy - ys[j]
            d = ddx * ddx + ddy * ddy  # squared distance (monotone, exact)
            if best_d is None or d < best_d or (d == best_d and j < best):
                best_d = d
                best = j
        visited[best] = True
        order.append(best)
        cx, cy = float(xs[best]), float(ys[best])
    length = 0.0
    px, py = float(dx), float(dy)
    for v in order:
        length += euclid(px, py, xs[v], ys[v])
        px, py = float(xs[v]), float(ys[v])
    length += euclid(px, py, float(dx), float(dy))
    prize = sum(pr[v] for v in order)
    return prize - length, length


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, dx, dy, xs, ys, pr = read_instance(sys.argv[1])

    ids = read_solution(sys.argv[2], n)
    if ids is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    P = tour_profit(ids, dx, dy, xs, ys, pr)
    P_base, D = visit_all_nn(n, dx, dy, xs, ys, pr)

    if D <= 0.0:
        # Degenerate (n == 0 or all nodes coincide with depot): full credit anchor.
        print(1_000_000)
        return

    score = 1_000_000.0 + 1_000_000.0 * (P - P_base) / D
    if score < 0.0:
        score = 0.0
    print(int(round(score)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic local scorer for "Continuous Point Placement then Snap".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is `k L` then `m` integer DEMAND points in [0,L]^2. A SOLUTION
    is `k` integer FACILITY coordinates, one pair per token-pair, each in
    [0, L]^2. The solution is FEASIBLE iff:
      (a) the output parses as exactly 2*k integers;
      (b) every facility coordinate (x,y) satisfies 0 <= x <= L and 0 <= y <= L.
    Anything else -- parse error, wrong count, a coordinate out of range -- is
    INFEASIBLE and scores 0 (the feasibility floor).
  * ENERGY of a feasible placement P = {p_0..p_{k-1}} (lower is better) is
        E(P) = coverage(P) - LAMBDA * dispersion(P)
    where, with Euclidean distance dist(.,.),
        coverage(P)   = sum over demand d of   min_i dist(d, p_i)
                        (every demand served by its nearest facility -- small is
                         good, so facilities should sit near demand)
        dispersion(P) = sum over facility i of min_{j != i} dist(p_i, p_j)
                        (nearest-neighbour spacing -- large is good, so
                         facilities should be spread apart; hence subtracted)
    LAMBDA is a fixed weight (see LAMBDA below). For k == 1 the dispersion term
    is defined as 0 (no other facility).
  * REFERENCE: the UNIFORM-GRID placement. Put the k facilities on a near-square
    grid spanning [0,L]^2 (g = round(sqrt(k)) columns; cells laid out row-major,
    the first k grid nodes used), snapped to integers. This is always feasible
    and is the naive "ignore the demand, just spread evenly" placement. Its
    energy is E_ref, recomputed inside the scorer so the reference is
    reproducible and solver-independent.
  * SCORE:
        score = round(1_000_000 * E_ref / E_solver)   (feasible, E_solver > 0)
        score = 2_000_000                              (feasible, E_solver <= 0;
                                                        a generous full-credit cap)
        score = 0                                      (infeasible)
    A higher score is better. The uniform-grid reference scores ~1_000_000; a
    demand-aware, well-spread placement lowers E below E_ref and so scores
    strictly more; a feasible-but-worse placement scores less but stays positive.

The scorer is self-contained and deterministic: it recomputes the uniform-grid
reference itself, so the baseline is reproducible and solver-independent.
"""
import sys
import math

LAMBDA = 0.5  # weight of the dispersion (spread) reward in the energy


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    k = int(next(it))
    L = int(next(it))
    demand = []
    for t in it:
        x = int(t)
        y = int(next(it))
        demand.append((x, y))
    return k, L, demand


def read_solution(path, k, L):
    """Return a list of k (x,y) integer facility coords, or None on error."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != 2 * k:
        return None
    pts = []
    for i in range(k):
        try:
            x = int(toks[2 * i])
            y = int(toks[2 * i + 1])
        except ValueError:
            return None
        if x < 0 or x > L or y < 0 or y > L:
            return None  # coordinate out of range -> infeasible
        pts.append((x, y))
    return pts


def coverage(demand, pts):
    """Sum over demand of distance to its nearest facility."""
    total = 0.0
    for (dx, dy) in demand:
        best = None
        for (px, py) in pts:
            ddx = dx - px
            ddy = dy - py
            d2 = ddx * ddx + ddy * ddy
            if best is None or d2 < best:
                best = d2
        total += math.sqrt(best)
    return total


def dispersion(pts):
    """Sum over facilities of the distance to the nearest OTHER facility."""
    k = len(pts)
    if k <= 1:
        return 0.0
    total = 0.0
    for i in range(k):
        pxi, pyi = pts[i]
        best = None
        for j in range(k):
            if j == i:
                continue
            ddx = pxi - pts[j][0]
            ddy = pyi - pts[j][1]
            d2 = ddx * ddx + ddy * ddy
            if best is None or d2 < best:
                best = d2
        total += math.sqrt(best)
    return total


def energy(demand, pts):
    return coverage(demand, pts) - LAMBDA * dispersion(pts)


def grid_reference(k, L):
    """Uniform near-square grid placement of k integer points in [0,L]^2."""
    g = int(round(math.sqrt(k)))
    if g < 1:
        g = 1
    rows = (k + g - 1) // g  # rows needed so g*rows >= k
    pts = []
    for idx in range(k):
        r = idx // g
        c = idx % g
        # spread columns over [0,L] and rows over [0,L]
        if g == 1:
            x = L // 2
        else:
            x = int(round(c * L / (g - 1)))
        if rows == 1:
            y = L // 2
        else:
            y = int(round(r * L / (rows - 1)))
        x = min(max(x, 0), L)
        y = min(max(y, 0), L)
        pts.append((x, y))
    return pts


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    k, L, demand = read_instance(sys.argv[1])

    # reference (uniform grid) energy -- recomputed by the scorer
    ref_pts = grid_reference(k, L)
    e_ref = energy(demand, ref_pts)

    pts = read_solution(sys.argv[2], k, L)
    if pts is None:
        print(0)  # parse error / wrong count / out of range -> infeasible
        return

    e_sol = energy(demand, pts)
    if e_sol <= 0:
        # dispersion dominated coverage: extraordinarily good -> full-credit cap
        print(2_000_000)
        return

    # e_ref should be positive for these instances; guard anyway
    if e_ref <= 0:
        e_ref = 1.0
    score = int(round(1_000_000.0 * e_ref / e_sol))
    print(score)


if __name__ == "__main__":
    main()

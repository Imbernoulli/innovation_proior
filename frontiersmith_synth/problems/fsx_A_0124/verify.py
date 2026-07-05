#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for cold-chain depot dispersion.

Objective (maximize): F = minimum pairwise Euclidean distance among the n depots.
Feasibility: exactly n points, each inside the unit square [0,1]^2, none strictly
inside any warm-zone disk, all distinct.

Baseline B (built by the checker, always feasible on generated instances):
    the single-row layout  (x_i, 0.5), x_i=(i+0.5)/n  has min pairwise distance 1/n.
Maximization normalization:
    sc    = min(1000, 100 * F / max(1e-9, B))
    Ratio = sc / 1000
Reproducing the row scores Ratio = 0.1; a layout with 10x the row's min distance caps at 1.0.
"""
import sys, math

TOL = 1e-6


def fail(reason):
    print("Infeasible (%s). Ratio: 0.0" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    n = int(next(it))
    c = int(next(it))
    disks = []
    for _ in range(c):
        cx = float(next(it)); cy = float(next(it)); r = float(next(it))
        disks.append((cx, cy, r))
    return n, disks


def read_points(path, n):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("no output")
    if len(toks) < 2 * n:
        fail("expected %d coordinate pairs, got %d numbers" % (n, len(toks)))
    pts = []
    for i in range(n):
        try:
            x = float(toks[2 * i]); y = float(toks[2 * i + 1])
        except ValueError:
            fail("non-numeric coordinate")
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate")
        pts.append((x, y))
    return pts


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, disks = read_instance(inf)
    pts = read_points(outf, n)

    # containment in unit square
    for (x, y) in pts:
        if x < -TOL or x > 1 + TOL or y < -TOL or y > 1 + TOL:
            fail("depot outside unit square")

    # warm-zone avoidance (must not be strictly inside any disk)
    for (x, y) in pts:
        for (cx, cy, r) in disks:
            d = math.hypot(x - cx, y - cy)
            if d < r - TOL:
                fail("depot inside a warm zone")

    # min pairwise distance + distinctness
    F = float("inf")
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - pts[j][0], yi - pts[j][1])
            if d < F:
                F = d
    if F < TOL:
        fail("two depots coincide")

    B = 1.0 / n
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

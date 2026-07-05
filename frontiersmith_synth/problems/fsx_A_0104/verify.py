#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for greenhouse-pod dispersion.

Problem (Heilbronn-triangle variant, maximization):
    Place n monitoring pods in the unit square [0,1]^2.  The robustness of the
    triangulated climate interpolation is limited by the SMALLEST triangle formed
    by any three pods.  Maximize

        F = min over all triples {i,j,k} of  area(pod_i, pod_j, pod_k).

Feasibility (tolerance 1e-6): exactly n coordinate pairs, all finite, all inside
[0,1]^2, and no two pods coincident.  Any violation -> `Ratio: 0.0`.

Baseline B (built by the checker itself; always feasible & strictly positive for the
prime n emitted by gen.py) is the quadratic-residue point set
        P_i = ( i/n , (i*i mod n)/n ),   i = 0..n-1
whose minimum triangle area the checker computes exactly.  Maximization scoring:
        sc    = min(1000, 100 * F / max(1e-9, B))
        Ratio = sc / 1000
Reproducing the baseline scores Ratio = 0.1; a layout whose min triangle area is 10x
the baseline caps at Ratio = 1.0.  `ans` (argv[3]) is an unused placeholder.
"""
import sys
import math

TOL = 1e-6


def fail(reason):
    print("Infeasible (%s). Ratio: 0.0" % reason)
    sys.exit(0)


def read_n(path):
    toks = open(path).read().split()
    if not toks:
        print("bad instance")
        sys.exit(1)
    return int(toks[0])


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
            x = float(toks[2 * i])
            y = float(toks[2 * i + 1])
        except ValueError:
            fail("non-numeric coordinate")
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate")
        pts.append((x, y))
    return pts


def min_triangle_area(pts):
    n = len(pts)
    m = float("inf")
    for i in range(n):
        ax, ay = pts[i]
        for j in range(i + 1, n):
            bx, by = pts[j]
            ux, uy = bx - ax, by - ay
            for k in range(j + 1, n):
                cx, cy = pts[k]
                a = abs(ux * (cy - ay) - (cx - ax) * uy) * 0.5
                if a < m:
                    m = a
    return m


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n = read_n(inf)
    pts = read_points(outf, n)

    # containment in the unit square [0,1]^2
    for (x, y) in pts:
        if x < -TOL or x > 1 + TOL or y < -TOL or y > 1 + TOL:
            fail("pod outside the greenhouse [0,1]^2")

    # distinctness (coincident pods are not allowed)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            if math.hypot(xi - pts[j][0], yi - pts[j][1]) < 1e-7:
                fail("two pods coincide")

    F = min_triangle_area(pts)

    # internal baseline: quadratic-residue point set (feasible & positive for prime n)
    base = [(i / n, (i * i % n) / n) for i in range(n)]
    B = min_triangle_area(base)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.8e B=%.8e Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

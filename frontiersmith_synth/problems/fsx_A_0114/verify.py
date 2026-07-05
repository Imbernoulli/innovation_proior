#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the lunar-habitat
Heilbronn (maximize-minimum-triangle-area) extremal point-configuration problem.

Objective (maximize): F = minimum triangle area over ALL triples of the n modules.
Feasibility: exactly n points, each inside the triangular plot A-B-C (tolerance 1e-6),
all pairwise distinct, all coordinates finite.

Baseline B (built by the checker, always feasible on every generated instance):
    a regular n-gon of radius 0.55 * inradius, centred at the plot's incentre. It sits
    strictly inside the incircle (hence strictly inside the plot). Its minimum-triangle
    area is computed exactly and used as the normaliser.

Maximization normalization:
    sc    = min(1000, 100 * F / max(1e-9, B))
    Ratio = sc / 1000
Reproducing that small central ring scores Ratio = 0.1; a configuration whose minimum
triangle area is 10x the ring's caps at Ratio = 1.0.
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
    A = (float(next(it)), float(next(it)))
    B = (float(next(it)), float(next(it)))
    C = (float(next(it)), float(next(it)))
    return n, A, B, C


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


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def inside_triangle(P, A, B, C):
    # orientation-agnostic: P is inside iff it is on the same side (>= 0 up to tol)
    # of every directed edge, using the plot's own orientation.
    orient = cross(A[0], A[1], B[0], B[1], C[0], C[1])
    s = 1.0 if orient >= 0 else -1.0
    d1 = s * cross(A[0], A[1], B[0], B[1], P[0], P[1])
    d2 = s * cross(B[0], B[1], C[0], C[1], P[0], P[1])
    d3 = s * cross(C[0], C[1], A[0], A[1], P[0], P[1])
    return d1 >= -TOL and d2 >= -TOL and d3 >= -TOL


def incircle(A, B, C):
    a = math.hypot(B[0] - C[0], B[1] - C[1])   # opposite A
    b = math.hypot(C[0] - A[0], C[1] - A[1])   # opposite B
    c = math.hypot(A[0] - B[0], A[1] - B[1])   # opposite C
    per = a + b + c
    ix = (a * A[0] + b * B[0] + c * C[0]) / per
    iy = (a * A[1] + b * B[1] + c * C[1]) / per
    area = 0.5 * abs(cross(A[0], A[1], B[0], B[1], C[0], C[1]))
    r = 2.0 * area / per
    return (ix, iy), r


def baseline_min_area(n, A, B, C):
    (ix, iy), r = incircle(A, B, C)
    rad = 0.55 * r
    ring = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        ring.append((ix + rad * math.cos(th), iy + rad * math.sin(th)))
    m = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = tri_area(ring[i], ring[j], ring[k])
                if a < m:
                    m = a
    return m


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, A, B, C = read_instance(inf)
    pts = read_points(outf, n)

    # containment
    for P in pts:
        if not inside_triangle(P, A, B, C):
            fail("module outside the triangular plot")

    # distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < TOL:
                fail("two modules coincide")

    # objective: minimum triangle area over all triples
    F = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = tri_area(pts[i], pts[j], pts[k])
                if a < F:
                    F = a

    B0 = baseline_min_area(n, A, B, C)
    sc = min(1000.0, 100.0 * F / max(1e-9, B0))
    print("F=%.8f B=%.8f Ratio: %.6f" % (F, B0, sc / 1000.0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic checker for the coral-reef Heilbronn-in-a-square problem.

Usage: python3 verify.py <in> <out> <ans>   (ans is an ignored placeholder)

Reads N and the (fixed unit-square) reef plot from <in>, reads N station
coordinates from <out>, validates strict feasibility, then computes
    F = min over all C(N,3) triples of the spanned triangle area (0.5*|cross|).
Objective is MAXIMIZED. The internal baseline B is a regular N-gon inscribed in
the plot's incircle. Prints `Ratio: F/B*0.1` capped at 1.0.
"""
import sys, math


def fail(reason):
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def min_triangle_area(pts):
    n = len(pts)
    best = float("inf")
    for a in range(n):
        xa, ya = pts[a]
        for b in range(a + 1, n):
            xb, yb = pts[b]
            dx1 = xb - xa
            dy1 = yb - ya
            for c in range(b + 1, n):
                xc, yc = pts[c]
                cross = dx1 * (yc - ya) - dy1 * (xc - xa)
                area = 0.5 * abs(cross)
                if area < best:
                    best = area
    return best


def baseline_ngon(N):
    # Weak reference construction: cluster the N stations on a SMALL central
    # ring (regular N-gon), centre (0.5,0.5), radius 0.5/sqrt(5). Areas scale
    # with radius^2, so this concentric ring is a genuine feasible layout that
    # is clearly suboptimal (it wastes the reef plot's corners). Its min-area
    # triangle is 3 consecutive vertices (a thin sliver), giving a positive
    # baseline with ample headroom for full-plot layouts.
    cx, cy = 0.5, 0.5
    r = 0.5 / math.sqrt(5.0)
    pts = []
    for k in range(N):
        t = 2.0 * math.pi * k / N
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return min_triangle_area(pts)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    intoks = read_tokens(sys.argv[1])
    if not intoks:
        fail("empty instance")
    try:
        N = int(intoks[0])
    except ValueError:
        fail("bad N")
    if N < 3:
        fail("N too small")

    outoks = read_tokens(sys.argv[2])
    # Need exactly 2*N coordinate tokens.
    if len(outoks) != 2 * N:
        fail("expected exactly %d coordinate tokens, got %d" % (2 * N, len(outoks)))

    pts = []
    for i in range(N):
        try:
            x = float(outoks[2 * i])
            y = float(outoks[2 * i + 1])
        except ValueError:
            fail("non-numeric coordinate")
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate")
        pts.append((x, y))

    tol = 1e-6
    for (x, y) in pts:
        if x < -tol or x > 1.0 + tol or y < -tol or y > 1.0 + tol:
            fail("station outside reef plot [0,1]^2")

    F = min_triangle_area(pts)
    if not math.isfinite(F) or F < 0:
        fail("bad objective")

    B = baseline_ngon(N)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("N=%d F=%.9f B=%.9f" % (N, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

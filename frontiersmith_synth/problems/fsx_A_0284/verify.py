#!/usr/bin/env python3
# Deterministic checker for "Summit Ridge Resort" (format C, quality-metric).
# Heilbronn-in-the-unit-square variant: place N points in [0,1]^2 so as to
# MAXIMIZE the minimum area of any triangle spanned by three of the points.
#
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]; the harness greps the LAST Ratio.
import sys, math

TOL = 1e-6         # absolute containment tolerance
RING_R = 0.20      # baseline ring radius (fits inside the unit square)
CX, CY = 0.5, 0.5  # plot center


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    return int(toks[0])


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def min_triangle_area(pts):
    """Exact minimum area over all C(n,3) triples (O(n^3), n small)."""
    n = len(pts)
    best = float("inf")
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            for c in range(b + 1, n):
                ar = tri_area(pa, pb, pts[c])
                if ar < best:
                    best = ar
    return best


def ring_baseline(N):
    """Checker's own baseline: N points equally spaced on a circle of radius
    RING_R centered at (CX, CY); a regular N-gon. Returns its min triangle
    area (dominated by 3 consecutive vertices)."""
    pts = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N
        pts.append((CX + RING_R * math.cos(ang), CY + RING_R * math.sin(ang)))
    return min_triangle_area(pts)


def main():
    try:
        N = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")
    if N < 3:
        fail("degenerate instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != 2 * N:
        fail("expected exactly %d numbers (N points)" % (2 * N))

    pts = []
    for k in range(N):
        try:
            x = float(otoks[2 * k])
            y = float(otoks[2 * k + 1])
        except Exception:
            fail("bad point %d" % k)
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite point %d" % k)
        if not (-TOL <= x <= 1.0 + TOL and -TOL <= y <= 1.0 + TOL):
            fail("point %d outside plot" % k)
        pts.append((x, y))

    F = min_triangle_area(pts)

    B = ring_baseline(N)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.9f B=%.9f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

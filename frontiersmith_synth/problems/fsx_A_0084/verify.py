#!/usr/bin/env python3
# Deterministic checker for "Glacier Sensor Net" (format C, quality-metric).
# Heilbronn-in-a-triangle variant: place N points in a triangle so as to
# MAXIMIZE the minimum area of any triangle spanned by three of the points.
#
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]; the harness greps the LAST Ratio.
import sys, math

TOL = 1e-6  # relative barycentric tolerance for containment
SHRINK = 0.6  # baseline crams beacons onto a small ring (fraction of inradius)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_triangle(path):
    toks = open(path).read().split()
    N = int(toks[0])
    ax, ay = float(toks[1]), float(toks[2])
    bx, by = float(toks[3]), float(toks[4])
    cx, cy = float(toks[5]), float(toks[6])
    return N, (ax, ay), (bx, by), (cx, cy)


def cross(ox, oy, px, py, qx, qy):
    return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)


def tri_area(p, q, r):
    return 0.5 * abs(cross(p[0], p[1], q[0], q[1], r[0], r[1]))


def min_triangle_area(pts):
    """Exact minimum area over all C(n,3) triples (O(n^3), n small)."""
    n = len(pts)
    best = float("inf")
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            for c in range(b + 1, n):
                pc = pts[c]
                ar = tri_area(pa, pb, pc)
                if ar < best:
                    best = ar
    return best


def in_triangle(P, A, B, C, area2):
    """Barycentric containment test with relative tolerance."""
    # signed areas (twice) of sub-triangles; total = area2 (>0, CCW triangle)
    d1 = cross(A[0], A[1], B[0], B[1], P[0], P[1])
    d2 = cross(B[0], B[1], C[0], C[1], P[0], P[1])
    d3 = cross(C[0], C[1], A[0], A[1], P[0], P[1])
    lim = -TOL * area2
    return d1 >= lim and d2 >= lim and d3 >= lim


def incircle_regular_ngon(A, B, C, N):
    """Checker's own trivial baseline: N points equally spaced on the
    triangle's incircle (all strictly inside/tangent). Returns min area."""
    # side lengths opposite each vertex
    a = math.hypot(B[0] - C[0], B[1] - C[1])  # opposite A
    b = math.hypot(C[0] - A[0], C[1] - A[1])  # opposite B
    c = math.hypot(A[0] - B[0], A[1] - B[1])  # opposite C
    s = 0.5 * (a + b + c)
    area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
    r = SHRINK * area / s  # small ring, a fraction of the inradius
    ix = (a * A[0] + b * B[0] + c * C[0]) / (a + b + c)
    iy = (a * A[1] + b * B[1] + c * C[1]) / (a + b + c)
    pts = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N
        pts.append((ix + r * math.cos(ang), iy + r * math.sin(ang)))
    return min_triangle_area(pts)


def main():
    try:
        N, A, B, C = read_triangle(sys.argv[1])
    except Exception:
        fail("bad instance")

    area2 = cross(A[0], A[1], B[0], B[1], C[0], C[1])  # twice signed area, CCW>0
    if area2 <= 0:
        fail("degenerate instance triangle")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) < 2 * N:
        fail("expected %d points (2N numbers)" % N)

    pts = []
    for k in range(N):
        try:
            x = float(otoks[2 * k])
            y = float(otoks[2 * k + 1])
        except Exception:
            fail("bad point %d" % k)
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite point %d" % k)
        P = (x, y)
        if not in_triangle(P, A, B, C, area2):
            fail("point %d outside triangle" % k)
        pts.append(P)

    F = min_triangle_area(pts)

    # internal trivial baseline: regular N-gon on the incircle
    B_base = incircle_regular_ngon(A, B, C, N)
    if B_base <= 0:
        B_base = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B_base))
    print("F=%.9f B=%.9f Ratio: %.6f" % (F, B_base, sc / 1000.0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Deterministic checker for "Pandemic Contact Net" (format C, quality-metric).
# Heilbronn-in-the-unit-square variant: place N points in the unit square so as
# to MAXIMIZE the minimum area of any triangle spanned by three of the points.
#
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]; the harness greps the LAST Ratio.
import sys, math

TOL = 1e-6          # absolute containment tolerance
CX, CY = 0.5, 0.5   # hall centre
RBASE = 0.3         # baseline ring radius (a shrunk concentric ring)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    N = int(toks[0])
    corners = [(float(toks[1 + 2 * k]), float(toks[2 + 2 * k])) for k in range(4)]
    return N, corners


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
    """Checker's own trivial baseline: N points equally spaced on the shrunk
    concentric ring of radius RBASE about the hall centre."""
    pts = [(CX + RBASE * math.cos(2.0 * math.pi * k / N),
            CY + RBASE * math.sin(2.0 * math.pi * k / N)) for k in range(N)]
    return min_triangle_area(pts)


def main():
    try:
        N, corners = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")
    if N < 1:
        fail("bad N")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) < 2 * N:
        fail("expected %d positions (2N numbers)" % N)

    pts = []
    for k in range(N):
        try:
            x = float(otoks[2 * k]); y = float(otoks[2 * k + 1])
        except Exception:
            fail("bad position %d" % k)
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite position %d" % k)
        if not (-TOL <= x <= 1.0 + TOL and -TOL <= y <= 1.0 + TOL):
            fail("position %d outside hall" % k)
        pts.append((x, y))

    F = min_triangle_area(pts)

    B = ring_baseline(N)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.9f B=%.9f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

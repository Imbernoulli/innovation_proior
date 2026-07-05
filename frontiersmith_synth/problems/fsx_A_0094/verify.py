#!/usr/bin/env python3
# Deterministic checker for "Sparse Telescope Array: Baseline Diversity" (format C).
# Maximize the minimum triangle area over all triples of station coordinates
# placed inside the unit right triangle T = conv{(0,0),(1,0),(0,1)}.
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys
import math

TOL = 1e-6

# Incenter and inradius of the unit right triangle with legs of length 1.
INR = (2.0 - math.sqrt(2.0)) / 2.0   # ~0.292893
CX = INR
CY = INR
RINC = INR * 0.999                   # largest safely-inscribed concyclic radius
BASE_FACTOR = 3.0                    # baseline circle radius = RINC / sqrt(3)
R_BASE = RINC / math.sqrt(BASE_FACTOR)
PHASE = 0.1


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def inside_triangle(x, y):
    return (x >= -TOL) and (y >= -TOL) and (x + y <= 1.0 + TOL)


def min_triangle_area(pts):
    """Exact minimum triangle area over all C(n,3) triples (n small)."""
    n = len(pts)
    m = float("inf")
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            ax = xj - xi
            ay = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                a = abs(ax * (yk - yi) - ay * (xk - xi))
                if a < m:
                    m = a
    return m / 2.0


def baseline_points(N):
    """Checker's internal trivial construction: N stations equally spaced on a
    small circle (radius R_BASE) about the incenter. Concyclic => no 3 collinear
    => strictly positive minimum triangle area, and every point lies inside T."""
    pts = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N + PHASE
        pts.append((CX + R_BASE * math.cos(ang), CY + R_BASE * math.sin(ang)))
    return pts


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        N = int(itoks[0])
    except Exception:
        fail("bad instance")

    if N < 3:
        fail("degenerate instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    # Exactly 2*N floating-point coordinates expected (no count line).
    if len(otoks) != 2 * N:
        fail("expected %d numbers, got %d" % (2 * N, len(otoks)))

    pts = []
    for k in range(N):
        try:
            x = float(otoks[2 * k])
            y = float(otoks[2 * k + 1])
        except Exception:
            fail("bad coordinate at station %d" % k)
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate at station %d" % k)
        if not inside_triangle(x, y):
            fail("station %d outside reserve" % k)
        pts.append((x, y))

    F = min_triangle_area(pts)
    if not math.isfinite(F) or F < 0.0:
        fail("invalid objective")

    B = min_triangle_area(baseline_points(N))

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("N=%d F=%.8f B=%.8f Ratio: %.6f" % (N, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

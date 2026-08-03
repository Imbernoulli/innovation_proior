import sys
import math
from itertools import combinations

# Deterministic checker for the "wind tunnel sensor placement" (Heilbronn-in-a-square) problem.
#   Input  (<in>) : one integer n, the number of sensors.
#   Output (<out>): exactly 2*n real numbers = the (x, y) coordinates of the n sensors.
# Objective (MAX): the minimum triangle area over all triples of sensors.
# The checker builds its own trivial baseline B (a flat elliptical arc) and normalizes.

TOL = 1e-6          # coordinate containment tolerance for the unit square
DEG = 1e-12         # coincident-point tolerance

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def min_triangle_area(pts):
    """Exact minimum triangle area over all C(n,3) triples (0.5*|cross|)."""
    n = len(pts)
    if n < 3:
        return 0.0
    best = float("inf")
    for a, b, c in combinations(range(n), 3):
        ax, ay = pts[a]
        bx, by = pts[b]
        cx, cy = pts[c]
        area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * 0.5
        if area < best:
            best = area
    return best

def ellipse_points(n, b):
    """Trivial feasible construction: n points evenly spaced on an axis-aligned
    ellipse inscribed in the unit square (semi-axes 0.5 in x, b in y).
    No line meets an ellipse in 3 points, so every triple has positive area."""
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        pts.append((0.5 + 0.5 * math.cos(t), 0.5 + b * math.sin(t)))
    return pts

def main():
    # ---- read instance ----
    try:
        toks = open(sys.argv[1]).read().split()
        n = int(toks[0])
    except Exception:
        fail("bad input")
    if n < 3:
        fail("n<3")

    # ---- internal baseline B: a flat elliptical arc (b0 = 0.10) ----
    B = min_triangle_area(ellipse_points(n, 0.10))
    B = max(B, 1e-12)

    # ---- parse participant output: exactly 2*n reals ----
    try:
        raw = open(sys.argv[2]).read().split()
        vals = [float(x) for x in raw]
    except Exception:
        fail("parse")
    if len(vals) != 2 * n:
        fail("expected %d numbers, got %d" % (2 * n, len(vals)))

    pts = []
    for i in range(n):
        x = vals[2 * i]
        y = vals[2 * i + 1]
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate")
        if x < -TOL or x > 1.0 + TOL or y < -TOL or y > 1.0 + TOL:
            fail("sensor %d out of the unit square: (%r, %r)" % (i, x, y))
        pts.append((x, y))

    # ---- coincident-sensor check ----
    for a, b in combinations(range(n), 2):
        dx = pts[a][0] - pts[b][0]
        dy = pts[a][1] - pts[b][1]
        if dx * dx + dy * dy < DEG * DEG:
            fail("coincident sensors %d and %d" % (a, b))

    # ---- objective + normalized score ----
    F = min_triangle_area(pts)
    sc = min(1000.0, 100.0 * F / B)
    print("minArea=%.9g baseline=%.9g Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()

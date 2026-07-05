import sys
import math
from itertools import combinations

# Deterministic checker for the "reservoir dam network" gauge-placement problem,
# a Heilbronn-type extremal configuration inside the unit TRIANGLE.
#   Input  (<in>) : one integer n, the number of gauge stations.
#   Output (<out>): exactly 2*n real numbers = the (x, y) coordinates of the stations.
# Objective (MAX): the minimum triangle area over all C(n,3) triples of stations.
# The checker builds its own trivial baseline B (a thin inscribed ellipse) and normalizes.

TOL = 1e-6          # containment tolerance for the unit triangle
DEG = 1e-9          # coincident-point tolerance

# Baseline construction parameters (an axis-aligned ellipse centred at the
# triangle's incentre; every such ellipse lies fully inside the triangle).
CX = 1.0 / (2.0 + math.sqrt(2.0))   # incentre x = incentre y
R0 = 0.28                           # x semi-axis shared by the baseline family
B_BASE = 0.028                      # y semi-axis of the thin baseline arc

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
    """Trivial feasible construction: n points evenly spaced around an axis-aligned
    ellipse (semi-axes R0 in x, b in y) centred at the triangle's incentre. The
    ellipse lies strictly inside the triangle, and no line meets an ellipse in 3
    points, so every triple has positive area."""
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        pts.append((CX + R0 * math.cos(t), CX + b * math.sin(t)))
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

    # ---- internal baseline B: a thin inscribed elliptical arc ----
    B = min_triangle_area(ellipse_points(n, B_BASE))
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
        # unit-triangle containment: x >= 0, y >= 0, x + y <= 1
        if x < -TOL or y < -TOL or x + y > 1.0 + TOL:
            fail("station %d outside the unit triangle: (%r, %r)" % (i, x, y))
        pts.append((x, y))

    # ---- coincident-station check ----
    for a, b in combinations(range(n), 2):
        dx = pts[a][0] - pts[b][0]
        dy = pts[a][1] - pts[b][1]
        if dx * dx + dy * dy < DEG * DEG:
            fail("coincident stations %d and %d" % (a, b))

    # ---- objective + normalized score ----
    F = min_triangle_area(pts)
    sc = min(1000.0, 100.0 * F / B)
    print("minArea=%.9g baseline=%.9g Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic checker for the museum-gallery-tour uniform-spread problem.

Usage: python3 verify.py <in> <out> <ans>   (ans is an ignored placeholder)

Objective (maximize):  U = d_min / d_max
where d_min and d_max are the smallest / largest pairwise Euclidean distances over
the COMBINED point set  S = {fixed landmarks} u {free stations}.  U in (0,1]; larger
means a more uniform spread (no cramped pair, no marooned pair).

Feasibility (any violation -> "Ratio: 0.0"):
  * output has exactly n points, each two finite floats
  * every coordinate in [0,1] (tolerance 1e-9)
  * no two points of S coincide (all pairwise distances > 1e-6)

Scoring:  baseline B = U of the checker's own diagonal construction (landmarks +
free stations equally spaced on the gallery's main diagonal).  sc = min(1000,
100*U/max(1e-9,B)); prints Ratio = sc/1000  (trivial baseline -> 0.1).
"""
import sys
import math

TOL = 1e-9
COINC = 1e-6


def fail(msg):
    print("reason: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def uratio(pts):
    m = len(pts)
    dmin = float("inf")
    dmax = 0.0
    for i in range(m):
        xi, yi = pts[i]
        for j in range(i + 1, m):
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < dmin:
                dmin = d
            if d > dmax:
                dmax = d
    return dmin, dmax


def baseline_stations(n):
    P = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        x = 0.05 + 0.90 * t
        y = 0.05 + 0.90 * t
        P.append((x, y))
    return P


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def main():
    inp, outp = sys.argv[1], sys.argv[2]

    itok = read_tokens(inp)
    idx = 0
    n = int(itok[idx]); idx += 1
    k = int(itok[idx]); idx += 1
    land = []
    for _ in range(k):
        x = float(itok[idx]); y = float(itok[idx + 1]); idx += 2
        land.append((x, y))

    # ---- participant output ----
    otok = read_tokens(outp)
    if len(otok) != 2 * n:
        fail("expected exactly %d numbers (%d points), got %d" % (2 * n, n, len(otok)))
    stations = []
    for i in range(n):
        try:
            x = float(otok[2 * i]); y = float(otok[2 * i + 1])
        except ValueError:
            fail("non-numeric coordinate")
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite coordinate")
        if x < -TOL or x > 1.0 + TOL or y < -TOL or y > 1.0 + TOL:
            fail("station out of gallery bounds [0,1]^2")
        stations.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))

    S = land + stations
    dmin, dmax = uratio(S)
    if dmax <= 0.0:
        fail("degenerate: all points coincide")
    if dmin < COINC:
        fail("two stations/landmarks coincide (d_min < 1e-6)")

    U = dmin / dmax

    # internal baseline
    B_dmin, B_dmax = uratio(land + baseline_stations(n))
    B = B_dmin / B_dmax if B_dmax > 0 else 1e-9

    sc = min(1000.0, 100.0 * U / max(1e-9, B))
    print("U=%.6f B=%.6f dmin=%.6f dmax=%.6f" % (U, B, dmin, dmax))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

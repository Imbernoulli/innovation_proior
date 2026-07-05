#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic weighted-circle-packing checker.

Reads the instance (substation count n, weights w_i) and the participant's
placement (x_i y_i r_i per substation). Validates feasibility strictly; on ANY
violation prints `Ratio: 0.0`. Otherwise scores by the weighted coverage
F = sum_i w_i * r_i, normalised against the checker's own uniform-grid baseline B.

    F   = sum_i w_i * r_i
    B   = weighted coverage of a ceil(sqrt(n)) x ceil(sqrt(n)) uniform grid
          with every disk radius 1/(2k)   (all weights, equal radius)
    sc  = min(1000, 100 * F / max(1e-9, B))       # maximisation
    Ratio = sc / 1000                              # grid -> ~0.1, 10x-better caps at 1.0

Feasibility (tolerance 1e-6):
  * each disk fully inside the unit square:  r <= x <= 1-r,  r <= y <= 1-r
  * radii non-negative
  * pairwise non-overlap:  dist(center_i, center_j) >= r_i + r_j - 1e-6
"""
import sys, math

TOL = 1e-6


def fail(reason):
    print("%s Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    toks = open(inf).read().split()
    n = int(toks[0])
    w = [float(toks[1 + i]) for i in range(n)]

    try:
        vals = [float(x) for x in open(outf).read().split()]
    except Exception:
        fail("parse-error.")
    if len(vals) != 3 * n:
        fail("wrong number of values (expected %d, got %d)." % (3 * n, len(vals)))

    disks = [(vals[3 * i], vals[3 * i + 1], vals[3 * i + 2]) for i in range(n)]

    # radius non-negative + containment
    for (x, y, r) in disks:
        if r < -TOL:
            fail("negative radius.")
        rr = max(0.0, r)
        if x - rr < -TOL or y - rr < -TOL or x + rr > 1.0 + TOL or y + rr > 1.0 + TOL:
            fail("disk leaves the service region.")

    # pairwise non-overlap
    for i in range(n):
        xi, yi, ri = disks[i]
        for j in range(i + 1, n):
            xj, yj, rj = disks[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < ri + rj - TOL:
                fail("coverage disks overlap.")

    F = sum(w[i] * max(0.0, disks[i][2]) for i in range(n))

    # internal uniform-grid baseline
    k = int(math.ceil(math.sqrt(n)))
    rb = 1.0 / (2.0 * k)
    B = sum(w) * rb

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.8f B=%.8f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

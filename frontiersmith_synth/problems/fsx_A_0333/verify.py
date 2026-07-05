#!/usr/bin/env python3
# Deterministic checker for "Metro Catchment Packing" (format C, maximize sum of radii).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]; harness greps the LAST "Ratio:".
#
# Instance:  N , then N platform coordinates (x_i, y_i) in the unit square.
# Artifact:  N lines "cx cy r" -- one catchment disk per station, in station order.
# Feasibility (tol=1e-6):
#   * exactly N disks, all numbers finite, r >= -tol;
#   * disk i CONTAINS its platform:  (cx-x)^2+(cy-y)^2 <= r^2 + tol;
#   * disk inside unit square: cx-r>=-tol, cx+r<=1+tol, cy-r>=-tol, cy+r<=1+tol;
#   * pairwise non-overlap: dist(center_i,center_j) >= r_i+r_j - tol.
# Objective F = sum r_i (maximize).
# Baseline B = "platform-centred symmetric growth" (see below); trivial reproduces it.
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    N = int(toks[0])
    pts = []
    idx = 1
    for _ in range(N):
        x = float(toks[idx]); y = float(toks[idx + 1]); idx += 2
        pts.append((x, y))
    return N, pts


def baseline(N, pts):
    # Simplest feasible construction: one UNIFORM radius shared by every disk,
    # each centred on its platform. Feasible when the common radius is at most
    # half the smallest inter-platform distance (pairwise non-overlap) and at
    # most the smallest wall distance (containment). B = N * that radius.
    minb = min(min(x, 1.0 - x, y, 1.0 - y) for (x, y) in pts)
    mind = None
    for i in range(N):
        xi, yi = pts[i]
        for j in range(i + 1, N):
            xj, yj = pts[j]
            d = math.hypot(xi - xj, yi - yj)
            if mind is None or d < mind:
                mind = d
    if mind is None:
        mind = 2.0 * minb
    runi = min(0.5 * mind, minb)
    if runi < 0.0:
        runi = 0.0
    return N * runi


def main():
    try:
        N, pts = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) < 3 * N:
        fail("truncated: need %d numbers" % (3 * N))

    cx = []; cy = []; rs = []
    for k in range(N):
        try:
            x = float(otoks[3 * k])
            y = float(otoks[3 * k + 1])
            r = float(otoks[3 * k + 2])
        except Exception:
            fail("bad disk %d" % k)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite disk %d" % k)
        if r < -TOL:
            fail("negative radius %d" % k)
        if r < 0.0:
            r = 0.0
        # containment inside the unit square
        if x - r < -TOL or x + r > 1.0 + TOL or y - r < -TOL or y + r > 1.0 + TOL:
            fail("disk %d leaves the city" % k)
        # must cover its own platform
        px, py = pts[k]
        if (x - px) ** 2 + (y - py) ** 2 > r * r + TOL:
            fail("disk %d misses its platform" % k)
        cx.append(x); cy.append(y); rs.append(r)

    # pairwise non-overlap (O(N^2), N <= 26)
    for a in range(N):
        for b in range(a + 1, N):
            d = math.hypot(cx[a] - cx[b], cy[a] - cy[b])
            if d < rs[a] + rs[b] - TOL:
                fail("overlap %d,%d" % (a, b))

    F = sum(rs)
    B = baseline(N, pts)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

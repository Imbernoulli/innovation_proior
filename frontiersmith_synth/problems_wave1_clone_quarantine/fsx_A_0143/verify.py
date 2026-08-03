#!/usr/bin/env python3
# Deterministic checker for "Arena Control Zones" (format C, maximize sum of radii
# of non-overlapping disks inside a circular arena).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        N = int(itoks[0]); R = float(itoks[1])
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if not otoks:
        fail("empty output")

    try:
        M = int(otoks[0])
    except Exception:
        fail("bad M")

    if M < 0 or M > N:
        fail("M out of range")

    need = 1 + 3 * M
    if len(otoks) < need:
        fail("truncated zones")

    xs = []; ys = []; rs = []
    for k in range(M):
        try:
            x = float(otoks[1 + 3 * k])
            y = float(otoks[2 + 3 * k])
            r = float(otoks[3 + 3 * k])
        except Exception:
            fail("bad zone %d" % k)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            fail("non-finite zone %d" % k)
        if r < -TOL:
            fail("negative radius %d" % k)
        # containment inside the circular arena: dist(center, origin) + r <= R
        d0 = math.sqrt(x * x + y * y)
        if d0 + r > R + TOL:
            fail("zone %d leaves arena" % k)
        xs.append(x); ys.append(y); rs.append(r)

    # non-overlap (O(M^2), M small)
    for a in range(M):
        for b in range(a + 1, M):
            dx = xs[a] - xs[b]; dy = ys[a] - ys[b]
            d = math.sqrt(dx * dx + dy * dy)
            if d < rs[a] + rs[b] - TOL:
                fail("overlap %d,%d" % (a, b))

    F = sum(rs)

    # internal trivial baseline: N equal zones evenly spread along a diameter.
    # The largest equal radius for a row of N disks touching along a chord of the
    # arena is rho = R/N, so the baseline sum is B = N * (R/N) = R.
    B = R

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

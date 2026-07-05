#!/usr/bin/env python3
# Deterministic checker for Lunar Habitat Sensor Lattice (format C, minimize star discrepancy).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def star_discrepancy(pts, n):
    """Exact 2D star discrepancy over the induced corner grid.
    pts: list of (x,y) in [0,1]^2. O(n^3)."""
    xs = sorted(set([p[0] for p in pts] + [1.0]))
    ys = sorted(set([p[1] for p in pts] + [1.0]))
    best = 0.0
    for qx in xs:
        for qy in ys:
            V = qx * qy
            nc = 0
            no = 0
            for (x, y) in pts:
                if x <= qx and y <= qy:
                    nc += 1
                    if x < qx and y < qy:
                        no += 1
            dplus = nc / n - V
            dminus = V - no / n
            m = dplus if dplus > dminus else dminus
            if m > best:
                best = m
    return best


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        n = int(itoks[0])
        d = int(itoks[1])
    except Exception:
        fail("bad instance")

    if d != 2:
        fail("unsupported dimension")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = 2 * n
    if len(otoks) != need:
        fail("expected exactly %d numbers (%d points), got %d" % (need, n, len(otoks)))

    pts = []
    for k in range(n):
        try:
            x = float(otoks[2 * k])
            y = float(otoks[2 * k + 1])
        except Exception:
            fail("bad point %d" % k)
        if not (x == x and y == y and abs(x) != float("inf") and abs(y) != float("inf")):
            fail("non-finite point %d" % k)
        if x < -TOL or x > 1.0 + TOL or y < -TOL or y > 1.0 + TOL:
            fail("point %d outside [0,1]^2" % k)
        # clamp into the unit square
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if y < 0.0:
            y = 0.0
        elif y > 1.0:
            y = 1.0
        pts.append((x, y))

    F = star_discrepancy(pts, n)

    # internal trivial baseline: n-point diagonal set
    base = [((i + 0.5) / n, (i + 0.5) / n) for i in range(n)]
    B = star_discrepancy(base, n)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

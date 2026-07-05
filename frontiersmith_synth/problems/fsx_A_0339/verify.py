#!/usr/bin/env python3
# Deterministic checker for the Tide Pool Biodiversity Survey.
# Format C, quality-metric: minimize the EXACT 3D star discrepancy of a point set.
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]; the harness greps the LAST "Ratio:".
import sys

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def star_discrepancy_3d(pts, n):
    """Exact 3D star discrepancy over the induced anchor grid.

    D*_N = sup over anchored boxes [0,q) / [0,q] of | count/N - vol |.
    The supremum is attained on the grid whose per-axis candidate coordinates
    are the point coordinates together with 1.0 (the closed-box upper corner).
    For each anchor q we evaluate both the closed count (coords <= q, driving
    the d+ term) and the open count (coords < q, driving the d- term).
    O((n+1)^3 * n)."""
    xs = sorted(set([p[0] for p in pts] + [1.0]))
    ys = sorted(set([p[1] for p in pts] + [1.0]))
    zs = sorted(set([p[2] for p in pts] + [1.0]))
    best = 0.0
    for qx in xs:
        for qy in ys:
            for qz in zs:
                V = qx * qy * qz
                nc = 0  # closed: all coords <= q
                no = 0  # open:   all coords <  q
                for (x, y, z) in pts:
                    if x <= qx and y <= qy and z <= qz:
                        nc += 1
                        if x < qx and y < qy and z < qz:
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

    if d != 3:
        fail("unsupported dimension")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = 3 * n
    if len(otoks) != need:
        fail("expected exactly %d numbers (%d points x 3 coords), got %d"
             % (need, n, len(otoks)))

    pts = []
    for k in range(n):
        try:
            x = float(otoks[3 * k])
            y = float(otoks[3 * k + 1])
            z = float(otoks[3 * k + 2])
        except Exception:
            fail("bad point %d" % k)
        for c in (x, y, z):
            # reject nan / inf
            if not (c == c) or c == float("inf") or c == float("-inf"):
                fail("non-finite coordinate in point %d" % k)
            if c < -TOL or c > 1.0 + TOL:
                fail("point %d outside [0,1]^3" % k)
        # clamp tiny overshoots into the closed cube
        def clamp(c):
            if c < 0.0:
                return 0.0
            if c > 1.0:
                return 1.0
            return c
        pts.append((clamp(x), clamp(y), clamp(z)))

    F = star_discrepancy_3d(pts, n)

    # internal trivial baseline: the n-point space diagonal (all mass on a line).
    base = [((i + 0.5) / n, (i + 0.5) / n, (i + 0.5) / n) for i in range(n)]
    B = star_discrepancy_3d(base, n)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Deterministic checker for the Salmon Migration Ladder low-discrepancy COMPLETION
# problem (format C, minimize exact star discrepancy of the FULL camera set).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def star_discrepancy(pts, n):
    """Exact 2D star discrepancy over the induced corner grid. O(n^3)."""
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
    # ---- parse instance ----
    try:
        itoks = open(sys.argv[1]).read().split()
        M = int(itoks[0])
        k = int(itoks[1])
    except Exception:
        fail("bad instance")
    if M <= 0 or k < 0 or k >= M:
        fail("bad instance sizes")
    fixed = []
    need_fixed = 2 * k
    body = itoks[2:]
    if len(body) < need_fixed:
        fail("instance missing fixed points")
    for j in range(k):
        fx = float(body[2 * j])
        fy = float(body[2 * j + 1])
        fixed.append((fx, fy))

    m = M - k  # number of cameras the solver must place

    # ---- parse participant output ----
    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    need = 2 * m
    if len(otoks) != need:
        fail("expected exactly %d numbers (%d new cameras), got %d" % (need, m, len(otoks)))

    added = []
    for t in range(m):
        try:
            x = float(otoks[2 * t])
            y = float(otoks[2 * t + 1])
        except Exception:
            fail("bad camera %d" % t)
        if not (x == x and y == y and abs(x) != float("inf") and abs(y) != float("inf")):
            fail("non-finite camera %d" % t)
        if x < -TOL or x > 1.0 + TOL or y < -TOL or y > 1.0 + TOL:
            fail("camera %d outside [0,1]^2" % t)
        # clamp into the unit square
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if y < 0.0:
            y = 0.0
        elif y > 1.0:
            y = 1.0
        added.append((x, y))

    # objective: star discrepancy of the FULL set (fixed + added), n = M
    full = fixed + added
    F = star_discrepancy(full, M)

    # internal trivial baseline B: complete the set by placing the m new cameras on
    # the main diagonal (concentrated -> deliberately poor spread).
    base_added = [((i + 0.5) / m, (i + 0.5) / m) for i in range(m)]
    base_full = fixed + base_added
    B = star_discrepancy(base_full, M)

    # minimization normalization
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

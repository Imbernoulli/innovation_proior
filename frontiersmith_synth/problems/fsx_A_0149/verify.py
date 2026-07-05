#!/usr/bin/env python3
# Deterministic checker for Geothermal Field Even Well Siting (format C, minimize D*).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
#
# Objective F = exact L-infinity star discrepancy of the emitted N-point set in [0,1]^2.
# The supremum defining D* is attained on the finite grid whose per-axis candidate
# coordinates are the well coordinates together with 1.0:
#   over-count (closed boxes):  max over grid  #{ p_j <= q (all axes) }/N - q_1*q_2
#   under-count (open  boxes):  max over grid  q_1*q_2 - #{ p_j <  q (all axes) }/N
#   D* = max(over, under).
import sys, math

TOL = 1e-6
EPS = 1e-9


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def star_discrepancy(xs, ys):
    """Exact 2D star discrepancy of points (xs[k], ys[k]) in [0,1]^2."""
    n = len(xs)
    cand_x = sorted(set(xs) | {1.0})
    cand_y = sorted(set(ys) | {1.0})
    dmax = 0.0
    for bx in cand_x:
        for by in cand_y:
            area = bx * by
            c_closed = 0
            c_open = 0
            for k in range(n):
                px = xs[k]; py = ys[k]
                if px <= bx + EPS and py <= by + EPS:
                    c_closed += 1
                if px < bx - EPS and py < by - EPS:
                    c_open += 1
            over = c_closed / n - area
            under = area - c_open / n
            if over > dmax:
                dmax = over
            if under > dmax:
                dmax = under
    return dmax


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        N = int(itoks[0]); D = int(itoks[1])
    except Exception:
        fail("bad instance")
    if D != 2:
        fail("unsupported dimension")

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

    if M != N:
        fail("M != N (must place all wells)")

    need = 1 + 2 * M
    if len(otoks) < need:
        fail("truncated wells")

    xs = []; ys = []
    for k in range(M):
        try:
            x = float(otoks[1 + 2 * k])
            y = float(otoks[2 + 2 * k])
        except Exception:
            fail("bad well %d" % k)
        if not (math.isfinite(x) and math.isfinite(y)):
            fail("non-finite well %d" % k)
        if x < -TOL or x > 1.0 + TOL or y < -TOL or y > 1.0 + TOL:
            fail("well %d outside field" % k)
        # clamp into [0,1] for the discrepancy math (within tolerance)
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        xs.append(x); ys.append(y)

    F = star_discrepancy(xs, ys)

    # internal trivial baseline: N wells evenly along the diagonal.
    bx = [(k + 0.5) / N for k in range(N)]
    B = star_discrepancy(bx, list(bx))

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

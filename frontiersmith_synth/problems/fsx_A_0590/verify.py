#!/usr/bin/env python3
# Deterministic checker for "Four-Bar Coupler Tracing" (format C, minimize).
#
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored)
# Instance: two fixed ground pivots + a target closed curve (sampled points).
# Participant output: 5 floats  a b c u v  = crank, coupler, rocker link
# lengths and the coupler-point offset (u along the coupler, v perpendicular).
# The checker sweeps the crank through a full turn, solves loop closure to
# trace the coupler point (trying both assembly branches, keeping the better),
# and scores a symmetric point-set (Chamfer) distance F to the target.
#
# Objective is MINIMIZE.  Baseline B = a fixed, un-tuned Grashof crank-rocker
# built by the checker from the ground span alone.  With F the participant's
# distance:  sc = min(1000, 100*B/F);  Ratio = sc/1000.
# Reproducing the baseline scores ~0.1; a 10x-closer trace caps at 1.0.
#
# Feasibility gate: exactly 5 finite tokens, a,b,c > 0.  Any violation -> 0.0.
# A linkage that cannot fully rotate leaves gaps in its trace; those gaps are
# penalized naturally by the target->trace half of the Chamfer distance.
import sys, math

K = 720  # crank samples used to trace the participant / baseline linkage


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    M = int(next(it))
    O0 = (float(next(it)), float(next(it)))
    O1 = (float(next(it)), float(next(it)))
    T = []
    for _ in range(M):
        x = float(next(it)); y = float(next(it))
        T.append((x, y))
    return M, O0, O1, T


def trace(O0, O1, a, b, c, u, v, branch):
    """Coupler points over a full crank turn (only reachable configs)."""
    ox, oy = O0
    o1x, o1y = O1
    pts = []
    for k in range(K):
        th = 2.0 * math.pi * k / K
        Ax = ox + a * math.cos(th)
        Ay = oy + a * math.sin(th)
        dx = o1x - Ax
        dy = o1y - Ay
        D = math.hypot(dx, dy)
        if D > b + c or D < abs(b - c) or D == 0.0:
            continue
        t = (b * b - c * c + D * D) / (2.0 * D)
        h2 = b * b - t * t
        if h2 < 0.0:
            continue
        h = math.sqrt(h2)
        fx = Ax + t * dx / D
        fy = Ay + t * dy / D
        ppx = -dy / D
        ppy = dx / D
        Bx = fx + branch * h * ppx
        By = fy + branch * h * ppy
        ex = Bx - Ax
        ey = By - Ay
        L = math.hypot(ex, ey)
        if L == 0.0:
            continue
        ex /= L
        ey /= L
        Px = Ax + u * ex + v * (-ey)
        Py = Ay + u * ey + v * (ex)
        pts.append((Px, Py))
    return pts


def chamfer(T, S):
    """Symmetric mean nearest-neighbour distance between point sets T and S."""
    if not S:
        return 1e18
    # target -> trace
    s1 = 0.0
    for (tx, ty) in T:
        best = 1e36
        for (sx, sy) in S:
            dd = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy)
            if dd < best:
                best = dd
        s1 += math.sqrt(best)
    s1 /= len(T)
    # trace -> target
    s2 = 0.0
    for (sx, sy) in S:
        best = 1e36
        for (tx, ty) in T:
            dd = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy)
            if dd < best:
                best = dd
        s2 += math.sqrt(best)
    s2 /= len(S)
    return 0.5 * (s1 + s2)


def best_distance(O0, O1, a, b, c, u, v, T):
    fp = chamfer(T, trace(O0, O1, a, b, c, u, v, +1.0))
    fm = chamfer(T, trace(O0, O1, a, b, c, u, v, -1.0))
    return min(fp, fm)


def main():
    try:
        M, O0, O1, T = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")
    if M < 3:
        fail("degenerate instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(otoks) < 5:
        fail("expected 5 numbers: a b c u v")
    try:
        a, b, c, u, v = (float(otoks[j]) for j in range(5))
    except Exception:
        fail("non-numeric output")
    for val in (a, b, c, u, v):
        if not math.isfinite(val):
            fail("non-finite parameter")
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        fail("link lengths must be positive")
    # sanity bound: reject absurd magnitudes (keeps trace meaningful & fast)
    g = math.hypot(O1[0] - O0[0], O1[1] - O0[1])
    if max(a, b, c, abs(u), abs(v)) > 1e4 * max(1.0, g):
        fail("parameters out of range")

    F = best_distance(O0, O1, a, b, c, u, v, T)

    # checker's own baseline: a fixed un-tuned Grashof crank-rocker from g alone
    a0, b0, c0 = 0.35 * g, 1.10 * g, 0.90 * g
    u0, v0 = 0.50 * b0, 0.30 * b0
    B = best_distance(O0, O1, a0, b0, c0, u0, v0, T)
    if B <= 0.0:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-12, F))
    print("F=%.9f B=%.9f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

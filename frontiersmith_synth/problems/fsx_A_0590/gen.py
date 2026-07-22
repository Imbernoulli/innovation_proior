import sys, math, random

# ---------------------------------------------------------------------------
# Four-bar coupler tracing (format C, quality-metric, minimize).
#
# `python3 gen.py <testId>` prints ONE instance:
#     M                       number of target curve samples
#     O0x O0y                 ground pivot 0 (fixed)
#     O1x O1y                 ground pivot 1 (fixed)
#     x y      (M lines)      target closed curve, sampled points
#
# The target is the coupler curve of a HIDDEN Grashof crank-rocker anchored at
# the two ground pivots, plus a small deterministic perturbation (so no linkage
# reproduces it exactly -> score headroom stays open).  All hidden linkages are
# genuine crank-rockers (crank fully rotates); the coupler point traces a
# closed algebraic sextic.  Seeded by testId only.
# ---------------------------------------------------------------------------

# Each hidden linkage: crank a, coupler b, rocker c, coupler-point offset (u,v)
# in the coupler frame; ground link g = |O1-O0| with O0=(0,0), O1=(g,0).
# df = perturbation magnitude as a fraction of g.  M = target sample count.
LADDER = [
    dict(g=10.0, a=2.0, b=8.0,  c=7.0,  u=3.0, v=3.5, M=90,  df=0.055),  # 1 small
    dict(g=10.0, a=2.2, b=7.5,  c=8.0,  u=2.5, v=4.5, M=110, df=0.055),  # 2
    dict(g=12.0, a=2.5, b=9.0,  c=8.5,  u=4.0, v=5.0, M=120, df=0.055),  # 3 trap (large v)
    dict(g=9.0,  a=1.8, b=7.0,  c=6.5,  u=2.0, v=4.8, M=130, df=0.055),  # 4 trap
    dict(g=11.0, a=2.0, b=8.5,  c=7.0,  u=5.0, v=5.5, M=140, df=0.055),  # 5 trap
    dict(g=13.0, a=2.6, b=10.0, c=9.0,  u=3.5, v=6.0, M=150, df=0.055),  # 6 trap
    dict(g=10.0, a=2.4, b=7.8,  c=8.2,  u=6.0, v=3.0, M=160, df=0.055),  # 7
    dict(g=14.0, a=3.0, b=11.0, c=10.0, u=4.5, v=6.5, M=170, df=0.055),  # 8 trap
    dict(g=12.0, a=2.3, b=9.5,  c=8.0,  u=2.8, v=6.8, M=185, df=0.055),  # 9 trap
    dict(g=15.0, a=3.2, b=12.0, c=11.0, u=5.5, v=7.0, M=200, df=0.055),  # 10 trap large
]


def trace(g, a, b, c, u, v, M, branch=1):
    """Sweep the crank through a full turn, solve loop closure, return the
    coupler point at each of M crank angles (ground truth uses branch=+1)."""
    pts = []
    O1x, O1y = g, 0.0
    for k in range(M):
        th = 2.0 * math.pi * k / M
        Ax = a * math.cos(th)
        Ay = a * math.sin(th)
        dx = O1x - Ax
        dy = O1y - Ay
        D = math.hypot(dx, dy)
        if D > b + c or D < abs(b - c) or D == 0.0:
            continue  # unreachable (does not happen for a Grashof crank-rocker)
        t = (b * b - c * c + D * D) / (2.0 * D)
        h = math.sqrt(max(0.0, b * b - t * t))
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
        # coupler point = A + u*(along AB) + v*(perp, +90 deg)
        Px = Ax + u * ex + v * (-ey)
        Py = Ay + u * ey + v * (ex)
        pts.append((Px, Py))
    return pts


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    d = LADDER[idx]
    g, a, b, c, u, v, M, df = d["g"], d["a"], d["b"], d["c"], d["u"], d["v"], d["M"], d["df"]

    pts = trace(g, a, b, c, u, v, M, branch=1)

    rng = random.Random(1000 + i)
    delta = df * g
    out = []
    for (x, y) in pts:
        ang = rng.uniform(0.0, 2.0 * math.pi)
        mag = delta * (0.5 + 0.5 * rng.random())
        out.append((x + mag * math.cos(ang), y + mag * math.sin(ang)))

    w = sys.stdout.write
    w("%d\n" % len(out))
    w("0 0\n")
    w("%.10g 0\n" % g)
    for (x, y) in out:
        w("%.10f %.10f\n" % (x, y))


if __name__ == "__main__":
    main()

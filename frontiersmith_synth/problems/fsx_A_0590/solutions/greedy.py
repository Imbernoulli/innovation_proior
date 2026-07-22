# TIER: greedy
# The obvious approach: blind local optimization of all five linkage
# parameters (a,b,c,u,v) to minimize the point-set distance, from a handful of
# generic random restarts.  It ignores the algebraic structure of the coupler
# curve, so it wanders into shallow local minima (and often near-degenerate,
# barely-rotating linkages).  Deterministic (fixed seed).
import sys, math
import numpy as np

toks = sys.stdin.read().split()
it = iter(toks)
M = int(next(it))
O0 = np.array([float(next(it)), float(next(it))])
O1 = np.array([float(next(it)), float(next(it))])
T = np.array([[float(next(it)), float(next(it))] for _ in range(M)])
g = float(np.hypot(*(O1 - O0)))

K = 180
TH = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
COS, SIN = np.cos(TH), np.sin(TH)


def trace(p, branch):
    a, b, c, u, v = p
    if a <= 0 or b <= 0 or c <= 0:
        return None
    Ax = O0[0] + a * COS
    Ay = O0[1] + a * SIN
    dx = O1[0] - Ax
    dy = O1[1] - Ay
    D = np.hypot(dx, dy)
    ok = (D <= b + c) & (D >= abs(b - c)) & (D > 0)
    if not ok.any():
        return None
    Dc = np.where(D > 0, D, 1.0)
    t = (b * b - c * c + D * D) / (2.0 * Dc)
    h2 = b * b - t * t
    ok = ok & (h2 >= 0)
    h = np.sqrt(np.clip(h2, 0, None))
    fx = Ax + t * dx / Dc
    fy = Ay + t * dy / Dc
    ppx = -dy / Dc
    ppy = dx / Dc
    Bx = fx + branch * h * ppx
    By = fy + branch * h * ppy
    ex = Bx - Ax
    ey = By - Ay
    L = np.hypot(ex, ey)
    L = np.where(L > 0, L, 1.0)
    ex /= L
    ey /= L
    Px = Ax + u * ex + v * (-ey)
    Py = Ay + u * ey + v * ex
    P = np.stack([Px, Py], 1)
    return P[ok]


def chamfer(S):
    if S is None or len(S) == 0:
        return 1e18
    d = np.sqrt(((T[:, None, :] - S[None, :, :]) ** 2).sum(2))
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def cost(p):
    return min(chamfer(trace(p, +1.0)), chamfer(trace(p, -1.0)))


# The naive recipe (the addendum's archetype): uniform random sampling of link
# lengths + a short local repair on the best draw.  Blind to the coupler
# curve's radial structure, most draws are non-Grashof or trace the wrong loop,
# and the short repair only reaches the nearest shallow local minimum.
rng = np.random.RandomState(20240607)
best_p, best_c = None, 1e18
for _ in range(400):
    p = np.array([
        rng.uniform(0.15, 0.8) * g,
        rng.uniform(0.5, 1.6) * g,
        rng.uniform(0.5, 1.3) * g,
        rng.uniform(-0.3, 1.0) * g,
        rng.uniform(-0.8, 0.8) * g,
    ])
    cc = cost(p)
    if cc < best_c:
        best_c, best_p = cc, p.copy()

# short local repair on the best random draw
p, cur = best_p.copy(), best_c
step = 0.08 * g
for _ in range(40):
    cand = p + rng.normal(0.0, 1.0, 5) * step
    if cand[0] > 0 and cand[1] > 0 and cand[2] > 0:
        cc = cost(cand)
        if cc < cur:
            p, cur = cand, cc
            continue
    step *= 0.97

print("%.10f %.10f %.10f %.10f %.10f" % tuple(p))

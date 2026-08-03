# TIER: strong
# Bottleneck-driven local search. Seed from the FULL inscribed ring (radius 0.5)
# and from random restarts; repeatedly find the smallest-area triple and push
# one of its vertices perpendicular to the opposite edge, keeping any move that
# increases the GLOBAL minimum triangle area. Deterministic.
import sys, math, random
from math import comb

CX, CY = 0.5, 0.5


def min_area_full(pts):
    """Return (min twice-area, bottleneck triple)."""
    n = len(pts)
    best = float("inf"); bt = (0, 1, 2)
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            bx0 = pb[0] - pa[0]; by0 = pb[1] - pa[1]
            for c in range(b + 1, n):
                pc = pts[c]
                ar = abs(bx0 * (pc[1] - pa[1]) - by0 * (pc[0] - pa[0]))
                if ar < best:
                    best = ar; bt = (a, b, c)
    return best, bt


def in_hall(P):
    return 0.0 <= P[0] <= 1.0 and 0.0 <= P[1] <= 1.0


def full_ring(N):
    r = 0.5
    return [(min(1.0, max(0.0, CX + r * math.cos(2 * math.pi * k / N))),
             min(1.0, max(0.0, CY + r * math.sin(2 * math.pi * k / N)))) for k in range(N)]


def optimize(N, rng, steps, init=None):
    pts = [list(p) for p in init] if init is not None else \
          [[rng.random(), rng.random()] for _ in range(N)]
    cur, bt = min_area_full(pts)
    step = 0.15
    for _ in range(steps):
        i, j, k = bt
        improved = False
        for (v, e0, e1) in ((i, j, k), (j, i, k), (k, i, j)):
            ex = pts[e1][0] - pts[e0][0]; ey = pts[e1][1] - pts[e0][1]
            L = math.hypot(ex, ey)
            nx, ny = (1.0, 0.0) if L < 1e-15 else (-ey / L, ex / L)
            for sgn in (1.0, -1.0):
                for mag in (step, 0.4 * step):
                    P = [pts[v][0] + sgn * mag * nx, pts[v][1] + sgn * mag * ny]
                    if not in_hall(P):
                        continue
                    old = pts[v]; pts[v] = P
                    nv, nbt = min_area_full(pts)
                    if nv > cur + 1e-15:
                        cur = nv; bt = nbt; improved = True
                        break
                    else:
                        pts[v] = old
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.6
            if step < 1e-4:
                break
    return cur, pts


def main():
    t = sys.stdin.read().split()
    N = int(t[0])
    trip = max(1, comb(N, 3))
    budget = 1_400_000
    steps = max(50, min(300, budget // (trip * 12)))
    restarts = 4
    best_val = -1.0; best_pts = None
    inits = [full_ring(N)] + [None] * restarts
    for s, init in enumerate(inits):
        rng = random.Random(9000 + 37 * s)
        val, pts = optimize(N, rng, steps, init=init)
        if val > best_val:
            best_val = val; best_pts = pts
    out = ["%.12f %.12f" % (p[0], p[1]) for p in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


main()

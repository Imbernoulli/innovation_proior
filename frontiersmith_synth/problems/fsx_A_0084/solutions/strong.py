# TIER: strong
# Bottleneck-driven local search: from several random seeds, repeatedly find
# the smallest-area triple and push one of its vertices to enlarge it, keeping
# any move that increases the GLOBAL minimum triangle area. Deterministic.
import sys, math, random


def read_tri():
    t = sys.stdin.read().split()
    N = int(t[0])
    A = (float(t[1]), float(t[2]))
    B = (float(t[3]), float(t[4]))
    C = (float(t[5]), float(t[6]))
    return N, A, B, C


def cross(ox, oy, px, py, qx, qy):
    return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)


def in_tri(P, A, B, C):
    d1 = cross(A[0], A[1], B[0], B[1], P[0], P[1])
    d2 = cross(B[0], B[1], C[0], C[1], P[0], P[1])
    d3 = cross(C[0], C[1], A[0], A[1], P[0], P[1])
    return d1 >= 0.0 and d2 >= 0.0 and d3 >= 0.0


def min_area_full(pts):
    """Return (minval, (i,j,k)) minimizing twice-area."""
    n = len(pts)
    best = float("inf"); bt = (0, 1, 2)
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            for c in range(b + 1, n):
                pc = pts[c]
                ar = abs(cross(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]))
                if ar < best:
                    best = ar; bt = (a, b, c)
    return best, bt


def sample(A, B, C, rng):
    u = rng.random(); v = rng.random()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    return (x, y)


def full_incircle(A, B, C, N):
    a = math.hypot(B[0] - C[0], B[1] - C[1])
    b = math.hypot(C[0] - A[0], C[1] - A[1])
    c = math.hypot(A[0] - B[0], A[1] - B[1])
    s = 0.5 * (a + b + c)
    area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
    r = area / s
    ix = (a * A[0] + b * B[0] + c * C[0]) / (a + b + c)
    iy = (a * A[1] + b * B[1] + c * C[1]) / (a + b + c)
    return [(ix + r * math.cos(2 * math.pi * k / N),
             iy + r * math.sin(2 * math.pi * k / N)) for k in range(N)]


def optimize(N, A, B, C, rng, steps, init=None):
    pts = list(init) if init is not None else [sample(A, B, C, rng) for _ in range(N)]
    cur, bt = min_area_full(pts)
    # characteristic length scale for step sizes
    scale = math.sqrt(abs(cross(A[0], A[1], B[0], B[1], C[0], C[1])))
    step = 0.15 * scale
    for _ in range(steps):
        i, j, k = bt
        improved = False
        # try pushing each bottleneck vertex perpendicular to the opposite edge
        trips = [(i, j, k), (j, i, k), (k, i, j)]
        for (v, e0, e1) in trips:
            ex = pts[e1][0] - pts[e0][0]
            ey = pts[e1][1] - pts[e0][1]
            L = math.hypot(ex, ey)
            if L < 1e-15:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = -ey / L, ex / L  # unit normal to opposite edge
            for sgn in (1.0, -1.0):
                for mag in (step, 0.4 * step):
                    px = pts[v][0] + sgn * mag * nx
                    py = pts[v][1] + sgn * mag * ny
                    P = (px, py)
                    if not in_tri(P, A, B, C):
                        continue
                    old = pts[v]
                    pts[v] = P
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
            if step < 1e-4 * scale:
                break
    return cur, pts


def main():
    N, A, B, C = read_tri()
    # budget: keep total O(steps * n^3) bounded across the ladder
    from math import comb
    trip = max(1, comb(N, 3))
    budget = 2_600_000
    steps = max(60, min(500, budget // (trip * 12)))
    restarts = 5
    best_val = -1.0; best_pts = None
    # seed 0: refine the full-inradius regular N-gon (guarantees >= greedy)
    inits = [full_incircle(A, B, C, N)] + [None] * restarts
    for s, init in enumerate(inits):
        rng = random.Random(9000 + 37 * s)
        val, pts = optimize(N, A, B, C, rng, steps, init=init)
        if val > best_val:
            best_val = val; best_pts = pts
    out = ["%.12f %.12f" % (p[0], p[1]) for p in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


main()

#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for redundant-arm path planning.

Validates feasibility (reach every waypoint, never collide a link with an
obstacle circle, joint angles bounded), computes the objective
F = total joint-space travel between consecutive waypoints (wrapped shortest
angular distance per joint, summed over joints and consecutive waypoint
pairs), builds the checker's own reference baseline B (an independent,
memory-less-but-obstacle-aware construction), and prints
`Ratio: <F->B normalized score>`.
"""
import sys, math

TOL = 0.03           # end-effector reach tolerance
COLL_EPS = 1e-6       # collision tolerance
THETA_BOUND = math.pi + 1e-6
ITERS_MAIN = 150
REFINE_ITERS = 90
PERTURB = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.2, -2.2]


def wrap_pi(a):
    while a > math.pi: a -= 2 * math.pi
    while a <= -math.pi: a += 2 * math.pi
    return a


def fk(theta, L):
    N = len(theta)
    x, y, phi = 0.0, 0.0, 0.0
    pts = [(0.0, 0.0)]
    for i in range(N):
        phi += theta[i]
        x += L[i] * math.cos(phi)
        y += L[i] * math.sin(phi)
        pts.append((x, y))
    return pts


def ccd(theta0, target, L, iters):
    theta = list(theta0)
    N = len(theta)
    for _ in range(iters):
        pts = fk(theta, L)
        for i in range(N, 0, -1):
            pivot = pts[i - 1]
            end = pts[N]
            ex, ey = end[0] - pivot[0], end[1] - pivot[1]
            tx, ty = target[0] - pivot[0], target[1] - pivot[1]
            if (ex == 0.0 and ey == 0.0) or (tx == 0.0 and ty == 0.0):
                continue
            delta = wrap_pi(math.atan2(ty, tx) - math.atan2(ey, ex))
            theta[i - 1] = wrap_pi(theta[i - 1] + delta)
            pts = fk(theta, L)
    return theta


def seg_point_dist(p0, p1, c):
    x0, y0 = p0; x1, y1 = p1; cx, cy = c
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-15:
        return math.hypot(cx - x0, cy - y0)
    t = ((cx - x0) * dx + (cy - y0) * dy) / L2
    t = max(0.0, min(1.0, t))
    px, py = x0 + t * dx, y0 + t * dy
    return math.hypot(cx - px, cy - py)


def min_clearance(theta, L, obstacles):
    if not obstacles:
        return 1e9
    pts = fk(theta, L)
    m = float('inf')
    for k in range(len(pts) - 1):
        for (cx, cy, r) in obstacles:
            d = seg_point_dist(pts[k], pts[k + 1], (cx, cy)) - r
            if d < m:
                m = d
    return m


def reach_err(theta, target, L):
    pts = fk(theta, L)
    return math.hypot(pts[-1][0] - target[0], pts[-1][1] - target[1])


def solve_waypoint(prev, target, L, obstacles):
    """Reference solver used for the checker's baseline B: primary CCD from
    `prev`, then (only if that collides) a fixed deterministic set of
    single-joint null-space perturbations, keeping the candidate with least
    continuity cost among the collision-free + tolerance-satisfying ones."""
    N = len(L)
    base = ccd(prev, target, L, ITERS_MAIN)
    if min_clearance(base, L, obstacles) >= 0 and reach_err(base, target, L) <= TOL:
        return base
    candidates = []
    for k in range(1, N - 1):
        for d in PERTURB:
            init = list(prev)
            init[k] = wrap_pi(init[k] + d)
            candidates.append(ccd(init, target, L, REFINE_ITERS))
    best, best_cost = None, None
    for th in candidates:
        if reach_err(th, target, L) > TOL:
            continue
        clr = min_clearance(th, L, obstacles)
        cont = sum(abs(wrap_pi(th[j] - prev[j])) for j in range(N))
        feas = clr >= 0
        cost = (0.0 if feas else (1e6 - clr * 1000.0)) + cont
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, th
    if best is not None and min_clearance(best, L, obstacles) >= 0:
        return best
    okcands = [th for th in candidates if reach_err(th, target, L) <= TOL]
    if okcands:
        return min(okcands, key=lambda th: -min_clearance(th, L, obstacles))
    return base


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    L = [float(next(it)) for _ in range(N)]
    targets = [(float(next(it)), float(next(it))) for _ in range(M)]
    obstacles = [(float(next(it)), float(next(it)), float(next(it))) for _ in range(K)]

    try:
        with open(outf) as f:
            otoks = f.read().split()
    except Exception:
        fail("could not read output")
        return

    expected = M * N
    if len(otoks) < expected:
        fail("too few numbers: expected %d got %d" % (expected, len(otoks)))
    if len(otoks) > expected:
        otoks = otoks[:expected]  # extra trailing tokens ignored (still validated below)

    vals = []
    for tok in otoks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
            return
        if not math.isfinite(v):
            fail("non-finite value")
            return
        vals.append(v)

    rows = [vals[i * N:(i + 1) * N] for i in range(M)]
    for row in rows:
        for th in row:
            if th < -THETA_BOUND or th > THETA_BOUND:
                fail("joint angle %.6f out of bounds [-pi,pi]" % th)
                return

    for i in range(M):
        err = reach_err(rows[i], targets[i], L)
        if err > TOL:
            fail("waypoint %d unreached: err=%.6f > tol=%.3f" % (i, err, TOL))
            return

    for i in range(M):
        clr = min_clearance(rows[i], L, obstacles)
        if clr < -COLL_EPS:
            fail("waypoint %d collides with obstacle (clearance=%.6f)" % (i, clr))
            return

    F = 0.0
    for i in range(M - 1):
        for j in range(N):
            F += abs(wrap_pi(rows[i + 1][j] - rows[i][j]))

    prev = [0.0] * N
    Brows = []
    for t in targets:
        th = solve_waypoint(prev, t, L, obstacles)
        Brows.append(th)
        prev = [0.0] * N  # baseline is memory-less: reset every waypoint
    B = 0.0
    for i in range(M - 1):
        for j in range(N):
            B += abs(wrap_pi(Brows[i + 1][j] - Brows[i][j]))
    if B <= 1e-9:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

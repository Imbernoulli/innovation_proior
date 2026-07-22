# TIER: trivial
"""Reproduces the checker's own reference baseline: solve each waypoint's
IK completely independently, always restarting CCD from the arm's neutral
rest pose (all joints = 0). It IS obstacle-aware (it will search for a
collision-free branch at a single waypoint) but has no memory across
waypoints at all, so consecutive configurations are unrelated -> large
joint-space path length."""
import sys, math

TOL = 0.03
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


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    L = [float(next(it)) for _ in range(N)]
    targets = [(float(next(it)), float(next(it))) for _ in range(M)]
    obstacles = [(float(next(it)), float(next(it)), float(next(it))) for _ in range(K)]

    lines = []
    for t in targets:
        th = solve_waypoint([0.0] * N, t, L, obstacles)  # always reset -> no memory
        lines.append(" ".join("%.12f" % v for v in th))
    sys.stdout.write("\n".join(lines) + "\n")


main()

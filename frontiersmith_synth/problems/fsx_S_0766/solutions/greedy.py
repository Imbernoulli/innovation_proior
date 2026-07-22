# TIER: greedy
"""The obvious "path-following IK" recipe: warm-start each waypoint's IK from
the PREVIOUS waypoint's solved joint angles (so consecutive configurations
stay close -- the standard trick for smooth trajectories) and run CCD to
convergence. This resolves the redundancy for continuity, but it never looks
at the obstacle list at all: it has no notion that the extra degrees of
freedom could also be used to dodge an obstacle, so on paths where the
"natural" CCD solution threads a link through an obstacle, it drives right
through it."""
import sys, math


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


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    L = [float(next(it)) for _ in range(N)]
    targets = [(float(next(it)), float(next(it))) for _ in range(M)]
    for _ in range(K):
        next(it); next(it); next(it)  # obstacles are read but ignored -- the trap

    ITERS_MAIN = 150
    prev = [0.0] * N
    lines = []
    for t in targets:
        th = ccd(prev, t, L, ITERS_MAIN)
        lines.append(" ".join("%.12f" % v for v in th))
        prev = th
    sys.stdout.write("\n".join(lines) + "\n")


main()

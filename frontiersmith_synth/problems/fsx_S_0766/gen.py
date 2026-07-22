#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE redundant-arm path-planning instance to stdout.

Deterministic: everything is a pure function of testId (no wall clock, no OS
randomness). testId 1..10 is a difficulty ladder; testId in {4,6,8,10} are
TRAP cases: an obstacle is planted exactly on the trajectory that a
"solve every waypoint independently from a fixed rest pose, ignore
obstacles" (obstacle-blind, memory-less) IK solver would trace, so that
approach collides, while a continuity/redundancy-aware solver can route
around it using the arm's spare degrees of freedom.
"""
import sys, math

# ---------------- shared kinematics (identical in verify.py / solutions) ----------------

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
    """Cyclic Coordinate Descent: iterate joints end->base aligning the
    end-effector direction with the target direction as seen from each pivot."""
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


def make_case(testId):
    N_tbl = {1: 5, 2: 5, 3: 6, 4: 6, 5: 6, 6: 7, 7: 7, 8: 7, 9: 8, 10: 8}
    M_tbl = {1: 10, 2: 11, 3: 11, 4: 12, 5: 13, 6: 13, 7: 14, 8: 15, 9: 16, 10: 18}
    K_tbl = {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 2, 9: 0, 10: 3}
    N, M, K = N_tbl[testId], M_tbl[testId], K_tbl[testId]
    ITERS_MAIN = 150

    L = [1.0 + 0.1 * (((i * 37 + testId * 13) % 7) - 3) / 3.0 for i in range(N)]
    Ltot = sum(L)

    A0 = 0.3 + 0.15 * testId
    A1 = 2.0 + 0.1 * (testId % 5)
    A2 = 0.5
    K1 = 1 + (testId % 3)
    R0 = 0.5 * Ltot
    R1 = 0.27 * Ltot
    K2 = 1 + ((testId + 1) % 2)

    targets = []
    for i in range(M):
        frac = i / (M - 1)
        ang = A0 + A1 * frac + A2 * math.sin(2 * math.pi * K1 * frac)
        rad = R0 + R1 * math.sin(2 * math.pi * K2 * frac + 0.7)
        targets.append((rad * math.cos(ang), rad * math.sin(ang)))

    obstacles = []
    if K > 0:
        prev = [0.0] * N
        traj = []
        for t in targets:
            th = ccd(prev, t, L, ITERS_MAIN)
            traj.append(th)
            prev = th
        avglen = sum(L) / N
        placed = 0
        widx_order = sorted(range(1, M - 1), key=lambda w: abs(w - M // 2))
        for widx in widx_order:
            if placed >= K:
                break
            for linkidx in range(0, N - 1):
                if placed >= K:
                    break
                for rfac in (0.30, 0.24, 0.18, 0.12):
                    pts = fk(traj[widx], L)
                    mx = 0.5 * (pts[linkidx][0] + pts[linkidx + 1][0])
                    my = 0.5 * (pts[linkidx][1] + pts[linkidx + 1][1])
                    r = rfac * avglen
                    if math.hypot(mx, my) < r + 0.65:
                        continue
                    bad = False
                    for (tx, ty) in targets:
                        if math.hypot(tx - mx, ty - my) < r + 0.08:
                            bad = True
                            break
                    if bad:
                        continue
                    for (ox, oy, orad) in obstacles:
                        if math.hypot(ox - mx, oy - my) < r + orad + 0.05:
                            bad = True
                            break
                    if bad:
                        continue
                    obstacles.append((mx, my, r))
                    placed += 1
                    break

    return N, M, K, L, targets, obstacles


def main():
    testId = int(sys.argv[1])
    N, M, K, L, targets, obstacles = make_case(testId)
    out = []
    out.append(f"{N} {M} {K}")
    out.append(" ".join("%.10f" % v for v in L))
    for (x, y) in targets:
        out.append("%.10f %.10f" % (x, y))
    for (cx, cy, r) in obstacles:
        out.append("%.10f %.10f %.10f" % (cx, cy, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

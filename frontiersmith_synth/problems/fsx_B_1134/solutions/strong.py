# TIER: strong
"""Insight: fold angles at a vertex flower are not 4 free numbers -- they live on a
1-dimensional curve of the loop-closure manifold (Rot(u1,theta1).R4(theta2,theta3,
theta4) == Identity), parametrized by the single free crease angle t = theta2. For
every candidate t we NEWTON-SOLVE the other three angles so the vertex never tears,
then search over t (the true, valid degree of freedom) for the value whose achieved
tip position is closest to the target -- instead of independently steering all four
raw crease angles as greedy does. Bridges (which carry no closure constraint) are
picked with the same closed-form axis fit greedy uses, but using the geometrically
correct (rotated) downstream offset.
"""
import sys, math


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def norm(a): return math.sqrt(max(0.0, dot(a, a)))
def normalize(a):
    n = norm(a)
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def matvec(R, v):
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


def matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def rodrigues(u, theta):
    ux, uy, uz = u
    c = math.cos(theta); s = math.sin(theta); C = 1.0 - c
    return (
        (c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s),
        (uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s),
        (uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C),
    )


def logmap(R):
    tr = R[0][0] + R[1][1] + R[2][2]
    cosang = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    ang = math.acos(cosang)
    if ang < 1e-8:
        return ((R[2][1] - R[1][2]) / 2.0, (R[0][2] - R[2][0]) / 2.0, (R[1][0] - R[0][1]) / 2.0)
    if ang > math.pi - 1e-6:
        x = math.sqrt(max(0.0, (R[0][0] + 1) / 2.0))
        y = math.sqrt(max(0.0, (R[1][1] + 1) / 2.0))
        z = math.sqrt(max(0.0, (R[2][2] + 1) / 2.0))
        axis = normalize((x, y, z))
        return scale(axis, ang)
    s = 2.0 * math.sin(ang)
    axis = ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s)
    return scale(axis, ang)


def _solve3(J, b):
    def det3(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det3(J)
    if abs(D) < 1e-14:
        return None
    x = [0.0, 0.0, 0.0]
    for k in range(3):
        Mk = [row[:] for row in J]
        for i in range(3):
            Mk[i][k] = b[i]
        x[k] = det3(Mk) / D
    return x


def solve_flower(us, t2, warm, iters=20, tol=1e-11):
    u1, u2, u3, u4 = us

    def resid(x):
        t1_, t3_, t4_ = x
        R2 = rodrigues(u2, t2)
        R3 = matmul(R2, rodrigues(u3, t3_))
        R4 = matmul(R3, rodrigues(u4, t4_))
        M = matmul(rodrigues(u1, t1_), R4)
        return logmap(M)

    x = list(warm)
    eps = 1e-6
    for _ in range(iters):
        r0 = resid(x)
        rn = math.sqrt(sum(c * c for c in r0))
        if rn < tol:
            break
        J = [[0.0] * 3 for _ in range(3)]
        for j in range(3):
            xp = list(x); xp[j] += eps
            xm = list(x); xm[j] -= eps
            rp = resid(xp); rm = resid(xm)
            for i in range(3):
                J[i][j] = (rp[i] - rm[i]) / (2 * eps)
        dx = _solve3(J, [-c for c in r0])
        if dx is None:
            break
        for i in range(3):
            x[i] += dx[i]
    return x[0], x[1], x[2], math.sqrt(sum(c * c for c in resid(x)))


def flower_geom(a, L):
    phis = (0.0, a[0], a[0] + a[1], a[0] + a[1] + a[2])
    us, pts = [], []
    for phi, Li in zip(phis, L):
        u = (math.cos(phi), math.sin(phi), 0.0)
        us.append(u)
        pts.append((Li * u[0], Li * u[1], Li * u[2]))
    return us, pts


def axis_fit_angle(p_axis, u, x, y):
    dx = sub(x, p_axis); dy = sub(y, p_axis)
    dxp = sub(dx, scale(u, dot(dx, u)))
    dyp = sub(dy, scale(u, dot(dy, u)))
    if norm(dxp) < 1e-9 or norm(dyp) < 1e-9:
        return 0.0
    c = dot(dxp, dyp)
    s = dot(cross(dxp, dyp), u)
    return math.atan2(s, c)


def best_t_for_flower(us, p3, target_local, tmax=1.25, ncoarse=241):
    """Grid + local refine search over the ONE free DOF t along the constraint curve,
    warm-started continuation, minimizing achieved-tip distance to target_local."""
    best = None  # (dist, t, th1, th3, th4)
    warm = (0.0, 0.0, 0.0)
    grid = [-tmax + 2 * tmax * i / (ncoarse - 1) for i in range(ncoarse)]
    # walk outward from t=0 in both directions so the continuation warm-start stays valid
    mid = ncoarse // 2
    order = list(range(mid, ncoarse)) + list(range(mid - 1, -1, -1))
    warm_pos, warm_neg = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    results = {}
    w = (0.0, 0.0, 0.0)
    for i in range(mid, ncoarse):
        t = grid[i]
        th1, th3, th4, rn = solve_flower(us, t, w)
        w = (th1, th3, th4)
        if rn < 1e-6:
            results[i] = (t, th1, th3, th4)
    w = (0.0, 0.0, 0.0)
    for i in range(mid - 1, -1, -1):
        t = grid[i]
        th1, th3, th4, rn = solve_flower(us, t, w)
        w = (th1, th3, th4)
        if rn < 1e-6:
            results[i] = (t, th1, th3, th4)

    for i, (t, th1, th3, th4) in results.items():
        R2 = rodrigues(us[1], t)
        R3 = matmul(R2, rodrigues(us[2], th3))
        tip = matvec(R3, p3)
        d = norm(sub(tip, target_local))
        if best is None or d < best[0]:
            best = (d, t, th1, th3, th4)

    # local refine: finer grid in a window around the best coarse point
    bi = None
    for i, (t, *_r) in results.items():
        if t == best[1]:
            bi = i
            break
    if bi is not None:
        lo = grid[max(0, bi - 2)]
        hi = grid[min(ncoarse - 1, bi + 2)]
        w = (best[2], best[3], best[4])
        nfine = 41
        for j in range(nfine):
            t = lo + (hi - lo) * j / (nfine - 1)
            th1, th3, th4, rn = solve_flower(us, t, w)
            if rn < 1e-6:
                w = (th1, th3, th4)
                R2 = rodrigues(us[1], t)
                R3 = matmul(R2, rodrigues(us[2], th3))
                tip = matvec(R3, p3)
                d = norm(sub(tip, target_local))
                if d < best[0]:
                    best = (d, t, th1, th3, th4)
    return best  # (dist, t, th1, th3, th4)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    K = int(next(it))
    modules = []
    for _ in range(K):
        a = [float(next(it)) for _ in range(4)]
        L = [float(next(it)) for _ in range(4)]
        modules.append({'a': a, 'L': L})
    targets = [tuple(float(next(it)) for _ in range(3)) for _ in range(K)]

    S_R, S_t = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)
    out_angles = []

    for k in range(K):
        a, L = modules[k]['a'], modules[k]['L']
        us, pts = flower_geom(a, L)
        p3 = pts[2]
        S_RT = tuple(zip(*S_R))
        target_local = matvec(S_RT, sub(targets[k], S_t))

        dist, t, th1, th3, th4 = best_t_for_flower(us, p3, target_local)
        out_angles += [th1, t, th3, th4]

        R2 = rodrigues(us[1], t)
        R3 = matmul(R2, rodrigues(us[2], th3))
        panel3_full_R = matmul(S_R, R3)
        panel3_full_t = S_t
        tip = add(panel3_full_t, matvec(panel3_full_R, p3))

        if k < K - 1:
            p4 = pts[3]
            w_local = normalize(sub(p4, p3))
            a2m, L2m = modules[k + 1]['a'], modules[k + 1]['L']
            us2, pts2 = flower_geom(a2m, L2m)
            p3_next = pts2[2]
            # correct (rotated) downstream offset -- unlike greedy's naive un-rotated add
            x_global = add(tip, matvec(panel3_full_R, p3_next))
            w_global = normalize(matvec(panel3_full_R, w_local))
            theta_b = axis_fit_angle(tip, w_global, x_global, targets[k + 1])
            out_angles.append(theta_b)
            Rb = rodrigues(w_local, theta_b)
            S_R = matmul(panel3_full_R, Rb)
            S_t = tip
        else:
            S_R, S_t = panel3_full_R, panel3_full_t

    print(" ".join("%.9f" % v for v in out_angles))


if __name__ == "__main__":
    main()

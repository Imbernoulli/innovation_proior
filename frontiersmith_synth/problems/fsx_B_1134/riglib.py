"""
riglib.py -- shared rigid-origami vertex-flower kinematics used by gen.py and verify.py.
NOT importable by solutions/ (sandboxed): solutions duplicate the small subset they need.

Model (see statement.md for the full description):
  A crease pattern is a chain of K degree-4 "vertex flowers" (modules). Module k has
  4 sector angles a1..a4 (Kawasaki: a1+a3=a2+a4=pi) and 4 crease lengths L1..L4. Its
  4 panels are triangles (v_k, p_i, p_{i+1}); crease_i is the edge (v_k, p_i), shared
  by panel_{i-1} and panel_i. crease_2,3,4 form a spanning tree of the flower's local
  panel-adjacency; crease_1 is the chord whose loop-closure equation
        Rot(u1, theta1) . R4(theta2,theta3,theta4)  ==  Identity
  (R4 = Rot(u2,theta2).Rot(u3,theta3).Rot(u4,theta4)) must hold for the flower to fold
  without tearing (rigid-foldability). Consecutive modules are joined by a hinge at
  module k's corner p3 (a fixed pivot -- module k+1's own local origin), with its own
  free "bridge" fold angle.
"""
import math

IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def norm(a): return math.sqrt(max(0.0, dot(a, a)))
def normalize(a):
    n = norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def matvec(R, v):
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


def matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def transpose(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))


def rodrigues(u, theta):
    ux, uy, uz = u
    c = math.cos(theta); s = math.sin(theta); C = 1.0 - c
    return (
        (c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s),
        (uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s),
        (uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C),
    )


def logmap(R):
    """SO(3) -> rotation vector (axis*angle), robust near 0 and near pi."""
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


def flower_geom(a, L):
    """Returns (us, pts): unit crease directions u1..u4 and flat 2D(z=0) corners p1..p4."""
    phis = (0.0, a[0], a[0] + a[1], a[0] + a[1] + a[2])
    us, pts = [], []
    for phi, Li in zip(phis, L):
        u = (math.cos(phi), math.sin(phi), 0.0)
        us.append(u)
        pts.append((Li * u[0], Li * u[1], Li * u[2]))
    return us, pts


def closure_residual_norm(u1, t1, R4):
    M = matmul(rodrigues(u1, t1), R4)
    return norm(logmap(M))


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


def solve_flower(us, t2, warm, iters=25, tol=1e-11):
    """Newton solve for (theta1,theta3,theta4) satisfying loop-closure given theta2=t2 fixed."""
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
    r_final = resid(x)
    rn = math.sqrt(sum(c * c for c in r_final))
    return x[0], x[1], x[2], rn


def solve_flower_path(us, t_target, nsteps=60):
    """Continuation from t=0 (trivial) to t=t_target; returns (theta1,theta3,theta4,resnorm)."""
    warm = (0.0, 0.0, 0.0)
    if nsteps <= 0:
        return solve_flower(us, t_target, warm)
    last = (0.0, 0.0, 0.0, 0.0)
    for i in range(1, nsteps + 1):
        tt = t_target * i / nsteps
        t1, t3, t4, rn = solve_flower(us, tt, warm)
        warm = (t1, t3, t4)
        last = (t1, t3, t4, rn)
    return last


def simulate_instance(modules, angles):
    """modules: list of {'a':[4],'L':[4]}. angles: flat list len 4K+(K-1).
    Returns (tips, closures): K tip 3D-points, K closure residual norms (radians)."""
    K = len(modules)
    S_R, S_t = IDENT, (0.0, 0.0, 0.0)
    tips, closures = [], []
    idx = 0
    for k in range(K):
        a = modules[k]['a']; L = modules[k]['L']
        us, pts = flower_geom(a, L)
        t1, t2, t3, t4 = angles[idx:idx + 4]
        idx += 4
        R2 = rodrigues(us[1], t2)
        R3 = matmul(R2, rodrigues(us[2], t3))
        R4 = matmul(R3, rodrigues(us[3], t4))
        closures.append(closure_residual_norm(us[0], t1, R4))
        panel3_full_R = matmul(S_R, R3)
        panel3_full_t = S_t
        p3 = pts[2]
        tip = add(panel3_full_t, matvec(panel3_full_R, p3))
        tips.append(tip)
        if k < K - 1:
            p4 = pts[3]
            w = normalize(sub(p4, p3))
            theta_b = angles[idx]; idx += 1
            Rb = rodrigues(w, theta_b)
            S_R = matmul(panel3_full_R, Rb)
            S_t = tip
    return tips, closures


def n_angles(K):
    return 4 * K + (K - 1)


def objective(modules, angles, targets, LAMBDA):
    tips, closures = simulate_instance(modules, angles)
    dist = sum(norm(sub(t, g)) for t, g in zip(tips, targets)) / len(tips)
    pen = sum(closures) / len(closures)
    return dist + LAMBDA * pen, dist, pen, tips, closures

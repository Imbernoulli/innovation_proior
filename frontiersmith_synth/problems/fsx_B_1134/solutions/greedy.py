# TIER: greedy
"""The obvious first idea: aim each crease at the target INDEPENDENTLY.

For each flower module, fit crease-2 and crease-3 each in isolation (a closed-form
single-axis "point onto target" rotation fit), holding the other at zero -- i.e.
treat the two hinges as if they moved the tip additively instead of by composed
rotation. Crease-1 and crease-4 don't move the tracked tip point at all under this
module's own local kinematics, so the greedy heuristic (which only ever asks "does
turning this crease move MY point closer to ITS target") leaves them at 0. Bridge
angles are fit similarly, also approximating the downstream sub-chain as a fixed
offset that is not re-rotated into the local frame.

This is a completely reasonable recipe a strong coder would write first -- and it
never asks whether the four crease angles at a vertex are jointly consistent
(loop-closure), so on any module where the target requires real folding it tears.
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


def axis_fit_angle(p_axis, u, x, y):
    """theta minimizing |Rot(p_axis,u,theta)(x) - y| (closed form: align perp components)."""
    dx = sub(x, p_axis); dy = sub(y, p_axis)
    dxp = sub(dx, scale(u, dot(dx, u)))
    dyp = sub(dy, scale(u, dot(dy, u)))
    if norm(dxp) < 1e-9 or norm(dyp) < 1e-9:
        return 0.0
    c = dot(dxp, dyp)
    s = dot(cross(dxp, dyp), u)
    theta = math.atan2(s, c)
    # a competent (if naive) coder still clips to the legal output range
    return max(-2.9, min(2.9, theta))


def flower_geom(a, L):
    phis = (0.0, a[0], a[0] + a[1], a[0] + a[1] + a[2])
    us, pts = [], []
    for phi, Li in zip(phis, L):
        u = (math.cos(phi), math.sin(phi), 0.0)
        us.append(u)
        pts.append((Li * u[0], Li * u[1], Li * u[2]))
    return us, pts


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

    ORIGIN = (0.0, 0.0, 0.0)
    S_R, S_t = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)
    out_angles = []

    for k in range(K):
        a, L = modules[k]['a'], modules[k]['L']
        us, pts = flower_geom(a, L)
        p3 = pts[2]
        target_local = matvec(tuple(zip(*S_R)), sub(targets[k], S_t))  # S_R^T (target - S_t)

        # independent single-axis fits, each pretending the OTHER hinge is at 0
        theta2 = axis_fit_angle(ORIGIN, us[1], p3, target_local)
        theta3 = axis_fit_angle(ORIGIN, us[2], p3, target_local)
        theta1 = 0.0
        theta4 = 0.0

        out_angles += [theta1, theta2, theta3, theta4]

        R2 = rodrigues(us[1], theta2)
        R3 = matmul(R2, rodrigues(us[2], theta3))
        panel3_full_R = matmul(S_R, R3)
        panel3_full_t = S_t
        tip = add(panel3_full_t, matvec(panel3_full_R, p3))

        if k < K - 1:
            p4 = pts[3]
            w_local = normalize(sub(p4, p3))
            # naive downstream approximation: add the next module's own flat offset
            # WITHOUT re-rotating it into the current local frame.
            a2m, L2m = modules[k + 1]['a'], modules[k + 1]['L']
            us2, pts2 = flower_geom(a2m, L2m)
            p3_next = pts2[2]
            x_global = add(tip, p3_next)
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

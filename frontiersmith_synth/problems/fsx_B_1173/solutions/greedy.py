# TIER: greedy
"""Textbook TDOA solve: (1) estimate each receiver's clock bias by averaging the
residual between observed and geometric TDOA over the calibration emitters (the
obvious way to handle the nuisance), then (2) for every held-out emitter, plug the
bias-corrected TDOA into the linear forward model and solve the resulting 2x2
normal-equations least-squares problem directly -- a single, uniform recipe applied
identically everywhere. It never looks at how well- or ill-conditioned a given
array's geometry is, so on the near-collinear "trap" arrays a tiny amount of
measurement noise gets amplified into a huge, useless position estimate."""
import sys
import math


def build_jacobian(receivers, x_ref, c):
    x0, y0 = receivers[0]
    d0 = math.hypot(x_ref[0] - x0, x_ref[1] - y0)
    u0 = ((x_ref[0] - x0) / d0, (x_ref[1] - y0) / d0)
    J = []
    for (xr, yr) in receivers[1:]:
        dr = math.hypot(x_ref[0] - xr, x_ref[1] - yr)
        ur = ((x_ref[0] - xr) / dr, (x_ref[1] - yr) / dr)
        J.append(((ur[0] - u0[0]) / c, (ur[1] - u0[1]) / c))
    return J


def solve_normal_eq(J, rhs):
    """Unregularized least-squares solve of J w = rhs (m x 2) via 2x2 normal eqns."""
    a11 = sum(j[0] * j[0] for j in J)
    a12 = sum(j[0] * j[1] for j in J)
    a22 = sum(j[1] * j[1] for j in J)
    b1 = sum(j[0] * r for j, r in zip(J, rhs))
    b2 = sum(j[1] * r for j, r in zip(J, rhs))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-18:
        det = 1e-18 if det >= 0 else -1e-18
    wx = (b1 * a22 - b2 * a12) / det
    wy = (a11 * b2 - a12 * b1) / det
    return wx, wy


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); R = int(next(it)); c = float(next(it))
    receivers = []
    for _ in range(R):
        x = float(next(it)); y = float(next(it))
        receivers.append((x, y))
    x_ref = (sum(p[0] for p in receivers) / R, sum(p[1] for p in receivers) / R)
    J = build_jacobian(receivers, x_ref, c)
    m = R - 1

    k_cal = int(next(it))
    bias_sum = [0.0] * m
    for _ in range(k_cal):
        px = float(next(it)); py = float(next(it))
        taus = [float(next(it)) for _ in range(m)]
        ox, oy = px - x_ref[0], py - x_ref[1]
        for i in range(m):
            pred = J[i][0] * ox + J[i][1] * oy
            bias_sum[i] += taus[i] - pred
    bias_hat = [b / k_cal for b in bias_sum]

    k_test = int(next(it))
    out = []
    for _ in range(k_test):
        taus = [float(next(it)) for _ in range(m)]
        corrected = [taus[i] - bias_hat[i] for i in range(m)]
        wx, wy = solve_normal_eq(J, corrected)
        out.append("%.6f %.6f" % (x_ref[0] + wx, x_ref[1] + wy))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

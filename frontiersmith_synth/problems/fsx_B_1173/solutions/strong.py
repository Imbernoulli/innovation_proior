# TIER: strong
"""Same bias correction as greedy, but before trusting the raw least-squares fit,
inspect the array's OWN 2x2 Gram matrix M = J^T J (a purely geometric quantity --
receiver positions only, no data needed) via its closed-form eigendecomposition.
This reports the array's uncertainty structure directly: one eigen-direction may
be well pinned down (large eigenvalue) while the other is nearly unobservable
(tiny eigenvalue, huge condition number) -- the geometric-dilution signature of a
near-collinear array.

For the well-constrained eigen-component, keep the ordinary least-squares fit
(same value greedy would produce). For the ill-constrained component (condition
number above a threshold), DISCARD the noise-dominated data-fit value and
substitute the calibration emitters' own centroid offset instead -- a prior
informed by real, already-paid-for measurements, not a guess. This is truncated-
SVD regularization in the array's natural eigenbasis: solve what the geometry
actually supports, regularize the rest toward the best available prior. On
well-conditioned arrays the truncation never fires and this is byte-for-byte
greedy; on the near-collinear trap arrays it avoids the noise blow-up entirely."""
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


def eigh_2x2(a11, a12, a22):
    """Closed-form eigendecomposition of the symmetric [[a11,a12],[a12,a22]].
    Returns (lam_lo, v_lo, lam_hi, v_hi), ascending eigenvalue order."""
    tr, diff = a11 + a22, a11 - a22
    disc = math.sqrt(max(0.0, diff * diff + 4 * a12 * a12))
    lam_hi = (tr + disc) / 2.0
    lam_lo = (tr - disc) / 2.0
    if abs(a12) > 1e-14:
        vh = (lam_hi - a22, a12)
    else:
        vh = (1.0, 0.0) if a11 >= a22 else (0.0, 1.0)
    nh = math.hypot(*vh)
    vh = (vh[0] / nh, vh[1] / nh)
    vl = (-vh[1], vh[0])
    return lam_lo, vl, lam_hi, vh


COND_THRESH = 1000.0


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
    cal_ox_sum = 0.0
    cal_oy_sum = 0.0
    for _ in range(k_cal):
        px = float(next(it)); py = float(next(it))
        taus = [float(next(it)) for _ in range(m)]
        ox, oy = px - x_ref[0], py - x_ref[1]
        cal_ox_sum += ox
        cal_oy_sum += oy
        for i in range(m):
            pred = J[i][0] * ox + J[i][1] * oy
            bias_sum[i] += taus[i] - pred
    bias_hat = [b / k_cal for b in bias_sum]
    prior = (cal_ox_sum / k_cal, cal_oy_sum / k_cal)

    # geometry-only Gram matrix and its eigenstructure -- fixed for the whole case
    a11 = sum(j[0] * j[0] for j in J)
    a12 = sum(j[0] * j[1] for j in J)
    a22 = sum(j[1] * j[1] for j in J)
    lam_lo, v_lo, lam_hi, v_hi = eigh_2x2(a11, a12, a22)
    lam_hi_safe = max(lam_hi, 1e-15)
    cond = lam_hi_safe / max(lam_lo, 1e-15)
    ill_conditioned = cond > COND_THRESH

    k_test = int(next(it))
    out = []
    for _ in range(k_test):
        taus = [float(next(it)) for _ in range(m)]
        corrected = [taus[i] - bias_hat[i] for i in range(m)]
        b1 = sum(j[0] * r for j, r in zip(J, corrected))
        b2 = sum(j[1] * r for j, r in zip(J, corrected))

        # well-constrained eigen-component: ordinary data fit
        c_hi = (v_hi[0] * b1 + v_hi[1] * b2) / lam_hi_safe
        # ill-constrained eigen-component: data fit, or the informed prior if the
        # geometry cannot support estimating it
        if ill_conditioned:
            c_lo = v_lo[0] * prior[0] + v_lo[1] * prior[1]
        else:
            c_lo = (v_lo[0] * b1 + v_lo[1] * b2) / max(lam_lo, 1e-15)

        wx = c_hi * v_hi[0] + c_lo * v_lo[0]
        wy = c_hi * v_hi[1] + c_lo * v_lo[1]
        out.append("%.6f %.6f" % (x_ref[0] + wx, x_ref[1] + wy))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

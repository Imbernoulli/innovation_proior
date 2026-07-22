#!/usr/bin/env python3
"""
gen.py <testId>  -- prints ONE training instance (payroll scroll fragment) to stdout.

Hidden model (never printed): a worker of wage w chooses hours h>=0 to maximize
    U(w,h) = w*h - T(w*h) - h^2/2
under a lost piecewise tax code with ONE bracket boundary z0:
    T(z) = tau_lo*z                                  for z <  z0
    T(z) = tau_lo*z0 + dT + tau_hi*(z-z0)             for z >= z0
(dT = a one-time surcharge assessed exactly at the boundary -- the "notch").

Training wages are drawn uniformly over [W_LO, W_HI], an archive window chosen (by
construction) to contain the ENTIRE bunching pile-up at z0 but to end exactly where
workers would first find it worthwhile to push through into the upper bracket -- so
NO training worker is ever observed inside the upper bracket. tau_lo, z0, tau_hi are
never printed; only N, W_LO, W_HI and the surcharge dT (a recovered scroll fragment)
are known facts, plus the (wage, hours) pairs themselves (with small idiosyncratic
noise, and -- in the harder tests -- a few corrupted/mis-transcribed entries).
"""
import sys, math, random

def true_params(t):
    rng = random.Random(100003 + t * 7919)
    tau_lo = rng.uniform(0.12, 0.24)
    gap = rng.uniform(0.12, 0.22)
    tau_hi = min(tau_lo + gap, 0.62)
    z0 = rng.uniform(500.0, 1500.0) * (1.0 + 0.10 * (t - 1))
    frac = rng.uniform(0.08, 0.16)          # target hole width as a fraction of z0
    dZ = frac * z0
    z_star = z0 + dZ
    dT = (1.0 - tau_hi) * dZ * dZ / (2.0 * z_star)
    w0_lo = math.sqrt(z0 / (1.0 - tau_lo))
    w0_hi = math.sqrt(z0 / (1.0 - tau_hi))
    w_star = math.sqrt(z_star / (1.0 - tau_hi))
    assert w_star >= w0_hi - 1e-9 >= w0_lo - 1e-9
    W_LO = w0_lo * 0.45
    W_HI = w_star
    N = 2500 + 250 * (t - 1)
    train_noise_sd = 0.010 + 0.003 * (t - 1)
    outlier_frac = 0.0 if t <= 5 else 0.008 + 0.004 * (t - 6)
    return dict(tau_lo=tau_lo, tau_hi=tau_hi, z0=z0, dT=dT, w0_lo=w0_lo, w0_hi=w0_hi,
                w_star=w_star, W_LO=W_LO, W_HI=W_HI, N=N,
                train_noise_sd=train_noise_sd, outlier_frac=outlier_frac)


def best_response_true(w, p):
    """Exact 3-candidate argmax best response under the TRUE schedule for scalar w>0."""
    tau_lo, tau_hi, z0, dT = p["tau_lo"], p["tau_hi"], p["z0"], p["dT"]
    w0_lo, w0_hi = p["w0_lo"], p["w0_hi"]
    best_u, best_h = -1e18, None
    if w <= w0_lo:
        h = w * (1.0 - tau_lo)
        u = (1.0 - tau_lo) ** 2 * w * w / 2.0
        if u > best_u:
            best_u, best_h = u, h
    # corner: always evaluable
    h_b = z0 / w
    u_b = (1.0 - tau_lo) * z0 - z0 * z0 / (2.0 * w * w)
    if u_b > best_u:
        best_u, best_h = u_b, h_b
    if w >= w0_hi:
        h_c = w * (1.0 - tau_hi)
        u_c = w * w * (1.0 - tau_hi) ** 2 / 2.0 - dT + z0 * (tau_hi - tau_lo)
        if u_c > best_u:
            best_u, best_h = u_c, h_c
    return best_h


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr); sys.exit(1)
    t = int(sys.argv[1])
    p = true_params(t)
    rng = random.Random(200003 + t * 7919)

    out = []
    out.append("%d %d %.6f %.6f %.6f" % (t, p["N"], p["W_LO"], p["W_HI"], p["dT"]))
    for _ in range(p["N"]):
        w = rng.uniform(p["W_LO"], p["W_HI"])
        h_true = best_response_true(w, p)
        h = h_true * (1.0 + rng.gauss(0.0, p["train_noise_sd"]))
        if rng.random() < p["outlier_frac"]:
            h = w * rng.uniform(0.05, 1.4)   # a garbled / mis-transcribed scroll entry
        h = max(h, 1e-4)
        out.append("%.6f %.6f" % (w, h))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

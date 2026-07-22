#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy hopper-discharge commissioning log to
stdout.

A hidden granular-discharge law relates the mass discharge rate Q (g/s) of a
flat-bottomed silo hopper to the aperture width D (cm), the grain diameter d
(cm), and the grain bulk density rho (g/cm^3):

    Q(D, d, rho) = C * rho * (D - k*d) ^ p        for D > k*d

  - rho enters EXACTLY linearly (mass flow is mass per volume times volume
    flow -- a dimensional necessity, not a fitted coefficient).
  - k*d is the empty-annulus OFFSET: grains near the aperture rim cannot
    reach the center of the opening, so the flow only "sees" an effective
    aperture (D - k*d), not the raw aperture D. k is a dimensionless
    multiple of the grain diameter (Beverloo-style).
  - p is a real-valued discharge exponent (near, but not fixed at, the
    classical 5/2 -- it depends on hopper cone angle / grain shape, which
    differ silo to silo).

The TRAIN commissioning log is what the solver SEES: small apertures,
D in [6,20] cm, with grain diameter d in [0.2,1.0] cm and bulk density rho
in [1.0,3.0] g/cm^3, all sampled independently. In that band D is only a
handful of grain-diameters wide, so the offset's curvature is real but easy
to under-read. The HELD-OUT grading grid (regenerated ONLY inside the
checker) covers LARGE apertures, D in [40,120] cm, with UNSEEN grain sizes
d in [1.2,3.2] cm -- never given to you.

STDOUT prints ONLY: header "<test_id> <n>" then n rows "<D> <d> <rho> <Q>".
The hidden law, its coefficients (C, k, p), and the seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
SEED_BASE = 896000
D_TR_LO, D_TR_HI = 6.0, 20.0
d_TR_LO, d_TR_HI = 0.2, 1.0
RHO_LO, RHO_HI = 1.0, 3.0
N_TRAIN = 200
NOISE_TRAIN = 0.02          # multiplicative lognormal sensor noise (train)


def params(t):
    """Hidden discharge law for this test id (identical in gen.py/verify.py)."""
    rng = random.Random(SEED_BASE + t * 104729)
    p_exp = rng.uniform(2.2, 2.9)      # discharge exponent
    k_off = rng.uniform(1.2, 2.4)      # offset, in grain diameters
    C = rng.uniform(0.4, 1.6)          # prefactor
    return p_exp, k_off, C


def true_Q(D, d, rho, p_exp, k_off, C):
    X = D - k_off * d
    if X <= 0.0:
        return None
    return C * rho * (X ** p_exp)


def gen_train(t):
    p_exp, k_off, C = params(t)
    rng = random.Random(11000 + t * 131)
    rows = []
    while len(rows) < N_TRAIN:
        D = rng.uniform(D_TR_LO, D_TR_HI)
        d = rng.uniform(d_TR_LO, d_TR_HI)
        rho = rng.uniform(RHO_LO, RHO_HI)
        q = true_Q(D, d, rho, p_exp, k_off, C)
        if q is None:
            continue
        q *= math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((D, d, rho, q))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (t, len(rows))]
    for D, d, rho, q in rows:
        out.append("%.8g %.8g %.8g %.8g" % (D, d, rho, q))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

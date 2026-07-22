#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy converter commissioning log to stdout.

A hidden power-loss law relates loss y (watts) to load fraction L and ambient
temperature T (deg C):

    y(L,T) = P0                                  standby loss (constant)
            + c1*L^2 + c2*T*L^2                   resistive loss w/ temp coef.
            + ks*L^p                              core-saturation loss

The TRAIN commissioning log is what the solver SEES: a bench test confined to
PARTIAL LOAD, L in [0.20, 0.60], with ambient temperature swept widely. In that
band the saturation term ks*L^p is small next to the resistive term (a few
percent near L=0.2, growing to ~15-20% by L=0.6) -- a faint but genuine
curvature signature, not zero. The HELD-OUT grading grid (regenerated ONLY
inside the grader) covers the OVERLOAD band L in [0.80, 1.10], where the same
term can exceed the resistive loss -- so a law that never resolved p diverges
there.

STDOUT prints ONLY: header "<n> <test_id>" then n rows "<L> <T> <y>". The
hidden law, its coefficients, and the seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
L_TR_LO, L_TR_HI = 0.20, 0.60
T_LO, T_HI = -10.0, 50.0
N_TRAIN = 200
NOISE_TRAIN = 0.025          # multiplicative lognormal sensor noise (train)


def params(t):
    """Hidden loss law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(500000 + t * 91013)
    p_exp = rng.uniform(3.4, 5.6)          # saturation exponent (non-integer)
    P0 = rng.uniform(3.0, 9.0)             # standby loss
    c1 = rng.uniform(45.0, 95.0)           # resistive coefficient at T=0
    beta = rng.uniform(-0.006, -0.002)     # relative temperature coefficient
    c2 = c1 * beta                         # coefficient of T*L^2
    ratio1 = rng.uniform(0.45, 0.85)       # saturation-term / resistive-term at L=1
    ks = ratio1 * c1
    return p_exp, P0, c1, c2, ks


def true_y(L, T, p_exp, P0, c1, c2, ks):
    return P0 + c1 * L * L + c2 * T * L * L + ks * L ** p_exp


def gen_train(t):
    p_exp, P0, c1, c2, ks = params(t)
    rng = random.Random(1000 + t * 13)
    rows = []
    for _ in range(N_TRAIN):
        L = rng.uniform(L_TR_LO, L_TR_HI)
        T = rng.uniform(T_LO, T_HI)
        y = true_y(L, T, p_exp, P0, c1, c2, ks) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((L, T, y))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for L, T, y in rows:
        out.append("%.8g %.8g %.8g" % (L, T, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

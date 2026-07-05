#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Wind-tunnel sensor calibration scaling law.  A pressure-tap sensor array is
calibrated against a reference flow field.  As two knobs are increased

    x1 = C  = sampling-compute budget   (number of averaged flow snapshots,
                                         spanning ~1.5 orders of magnitude)
    x2 = R  = sensor-grid resolution    (effective tap count)

the residual calibration error  L  decreases.  Empirically L obeys a scaling
law with an *irreducible instrument-noise floor*: no matter how much compute or
resolution you spend, L cannot fall below the physical floor.

Each test id fixes a DIFFERENT hidden scaling law.  The solver only ever sees
noisy TRAINING measurements taken in the CHEAP calibration regime (small C, R).
The graded split lives in a much LARGER-compute / higher-resolution
EXTRAPOLATION regime and is regenerated inside the grader only -- it is never
printed here.  Predicting L there requires having identified the floor, not just
a slope, from the cheap regime.

Difficulty ladder (testId 1..10): more measurement noise + fewer training rows.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows "C R L".
The hidden law and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    rng = random.Random(310007 + t * 6421)
    Einf = rng.uniform(0.50, 1.20)   # irreducible instrument-noise floor
    A = rng.uniform(1.5, 3.5)        # compute-limited amplitude
    alpha = rng.uniform(0.35, 0.65)  # compute scaling exponent (HIDDEN)
    B = rng.uniform(1.2, 3.0)        # resolution-limited amplitude
    beta = rng.uniform(0.35, 0.65)   # resolution scaling exponent (HIDDEN)
    return Einf, A, alpha, B, beta


def fval(C, R, cf):
    Einf, A, alpha, B, beta = cf
    return Einf + A * C ** (-alpha) + B * R ** (-beta)


# training regime (cheap): C,R log-uniform in [1, 25]
C_LO, C_HI = 1.0, 25.0
R_LO, R_HI = 1.0, 25.0


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.04 + (t - 1) * 0.015      # additive measurement noise
    n = 180 - (t - 1) * 12
    cf = coeffs(t)
    rng = random.Random(880 + t * 97711)
    out = ["%d %d" % (n, t)]
    lgC = math.log(C_LO), math.log(C_HI)
    lgR = math.log(R_LO), math.log(R_HI)
    for _ in range(n):
        C = math.exp(rng.uniform(*lgC))
        R = math.exp(rng.uniform(*lgR))
        L = fval(C, R, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r" % (C, R, L))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

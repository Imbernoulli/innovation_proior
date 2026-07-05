#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAINING sample to stdout.

Factory production recovery: a single plant's output y is governed by a fixed
but UNKNOWN constant-elasticity-of-substitution (CES) production law over two
normalised inputs
    K = capital (machine-hours, normalised)
    L = labour  (worker-hours, normalised)
The hidden law has the economics form
    y = A * ( delta*K^(-rho) + (1-delta)*L^(-rho) )^(-nu/rho)
with hidden total-factor productivity A, capital share delta, substitution
parameter rho (elasticity of substitution = 1/(1+rho), HIDDEN and away from the
Cobb-Douglas point rho=0), and returns-to-scale nu.  Output is measured with
multiplicative (log-normal) productivity noise.

During normal operation the plant only samples a bounded input region
K,L in [KLO,KHI].  The grader's HELD-OUT split lives in a HIGHER-throughput
EXTRAPOLATION region (larger K and L, outside the calibration box) and is
regenerated inside the grader only -- it is never printed here.

Difficulty ladder (testId 1..10): more measurement noise + fewer samples.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
"K L y".  The hidden law and its seed are NOT printed.
"""
import sys, random, math

# training-region input box (normal operating range).  Note K and L are drawn
# INDEPENDENTLY here, so the training input-ratio K/L already spans a wide range;
# the held-out split (in the grader) pushes the RATIO outside this range -- the
# regime where the elasticity of substitution actually bites.
KLO, KHI = 0.50, 1.50


def coeffs(t):
    """Hidden CES parameters for test id t (deterministic in t only)."""
    rng = random.Random(60420 + t * 7919)
    A = rng.uniform(0.90, 1.35)          # total-factor productivity
    delta = rng.uniform(0.35, 0.65)      # capital share
    # substitution parameter kept well AWAY from 0 (Cobb-Douglas is biased),
    # sign alternates across the ladder so per-test behaviour diverges.
    mag = rng.uniform(0.85, 2.10)
    rho = mag if (t % 2 == 1) else -min(mag, 0.75)   # rho > -1 keeps base>0
    nu = rng.uniform(0.85, 1.15)         # returns to scale
    return A, delta, rho, nu


def fval(K, L, cf):
    A, delta, rho, nu = cf
    base = delta * K ** (-rho) + (1.0 - delta) * L ** (-rho)
    return A * base ** (-nu / rho)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.03 + (t - 1) * 0.018       # log-normal productivity noise
    n = 220 - (t - 1) * 15
    cf = coeffs(t)
    rng = random.Random(500 + t * 104729)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        K = rng.uniform(KLO, KHI)
        L = rng.uniform(KLO, KHI)
        y = fval(K, L, cf) * math.exp(rng.gauss(0.0, sigma))
        out.append("%r %r %r" % (K, L, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

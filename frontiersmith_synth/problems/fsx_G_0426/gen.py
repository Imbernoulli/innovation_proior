#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Variable-star survey: a hidden period-luminosity-color-metallicity (PLZ)
relation maps three per-star observables
    x1 = log10(pulsation period / day)     x2 = mean color index (V-I)
    x3 = metallicity [Fe/H]
to the absolute magnitude y = M_V (brighter stars are MORE NEGATIVE).  Each
test id fixes a DIFFERENT hidden calibration; the solver only ever sees noisy
TRAINING measurements drawn from the well-sampled SHORT/MEDIUM-period regime
x1 in [0.0, 1.0]  (P from 1 to ~10 days).  The graded HELD-OUT split is the
LONG-PERIOD tail x1 in [1.0, 1.8] (P up to ~63 days), a genuine extrapolation
region regenerated only inside the grader -- it is never printed here.

Difficulty ladder (testId 1..10): more photometric noise + fewer stars.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
"x1 x2 x3 y".  The hidden relation and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    rng = random.Random(310007 + t * 6301)
    a = rng.uniform(-3.4, -2.4)    # log-period slope (brighter with longer P)
    q = rng.uniform(-0.70, -0.30)  # period-luminosity CURVATURE (nonlinear break)
    b = rng.uniform(0.80, 1.40)    # color term
    m = rng.uniform(0.15, 0.45)    # metallicity zero-point term
    d = rng.uniform(-0.40, -0.10)  # metallicity-DEPENDENT slope (Fe/H * logP)
    g = rng.uniform(-4.60, -3.80)  # absolute-magnitude zero point
    return a, q, b, m, d, g


def fval(x, cf):
    a, q, b, m, d, g = cf
    lp = x[0]
    return a * lp + q * lp * lp + b * x[1] + m * x[2] + d * x[2] * lp + g


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.05 + (t - 1) * 0.03
    n = 220 - (t - 1) * 16
    cf = coeffs(t)
    rng = random.Random(4400 + t * 99991)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0),    # log period (short/medium)
             rng.uniform(0.5, 1.2),    # color V-I
             rng.uniform(-1.5, 0.0)]   # metallicity
        y = fval(x, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r %r" % (x[0], x[1], x[2], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Smart-city adaptive street-lighting: a hidden per-lamp power law couples four
normalised sensor readings
    x1 = ambient-darkness index        x2 = pedestrian/traffic pressure
    x3 = wind exposure                 x4 = fixture ageing factor
to the measured luminaire power draw y.  Each test id fixes a DIFFERENT hidden
law; the solver only ever sees noisy TRAINING measurements drawn from the
normal daytime-calibration region x_i in [0,1].  The held-out grading split
lives in a heavier-load EXTRAPOLATION region and is regenerated inside the
grader only -- it is never printed here.

Difficulty ladder (testId 1..10): more measurement noise + fewer samples.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows.
The hidden law and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    rng = random.Random(90001 + t * 7919)
    a = rng.uniform(1.8, 3.2)     # x1^2  (darkness -> nonlinear boost)
    b = rng.uniform(2.5, 4.0)     # exp scale
    c = rng.uniform(0.45, 0.75)   # exp rate (HIDDEN, varies per lamp)
    d = rng.uniform(-2.0, -0.6)   # wind * ageing coupling
    e = rng.uniform(0.6, 1.4)     # x1 linear
    g = rng.uniform(-0.5, 0.5)    # offset
    return a, b, c, d, e, g


def fval(x, cf):
    a, b, c, d, e, g = cf
    return a * x[0] * x[0] + b * math.exp(c * x[1]) + d * x[2] * x[3] + e * x[0] + g


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.08 + (t - 1) * 0.05
    n = 200 - (t - 1) * 15
    cf = coeffs(t)
    rng = random.Random(500 + t * 104729)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]
        y = fval(x, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r %r %r" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

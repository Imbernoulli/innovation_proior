#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE outbreak-dashboard TRAINING sample to stdout.

Epidemiology / outbreak dashboard.  A public-health team watches a new
outbreak.  A hidden per-outbreak *incidence law* maps

    x1 = t          time in weeks since first detection
    x2 = stringency normalised non-pharmaceutical-intervention level in [0,1]
    x3 = testing    normalised surveillance / reporting intensity   in [0,1]
    x4 = holiday    a normalised calendar/mobility nuisance signal   in [0,1]

to the reported new-case incidence y.  The epidemic curve itself is a skewed
single-peak wave (fast exponential rise, slower exponential decay -- the
asymmetric "double-exponential" outbreak shape) plus small additive
surveillance effects.

The dashboard only exposes the EARLY window of the outbreak: the rise, the
peak, and the very start of the decline (t in [0, 11]).  The graders held-out
split is the POST-PEAK TAIL (t in [11, 17.5]) and is regenerated only inside
the grader -- it is never printed here.  Recovering the tail therefore means
extrapolating the decay law you can only glimpse in the training window.

Each test id fixes a DIFFERENT hidden outbreak.  Difficulty ladder
(testId 1..10): more measurement noise + fewer reported weeks.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
"x1 x2 x3 x4 y".  The hidden law, its seed and its coefficients are NOT printed.
"""
import sys, random, math

T0, T1 = 0.0, 11.0          # training window on x1 (early outbreak)


def coeffs(t):
    rng = random.Random(60413 + t * 7919)
    A  = rng.uniform(45.0, 85.0)     # peak-scaling amplitude (linear)
    a  = rng.uniform(0.60, 0.90)     # exponential RISE rate    (hidden)
    b  = rng.uniform(0.30, 0.48)     # exponential DECAY rate    (hidden, slower)
    tp = rng.uniform(6.0, 8.0)       # peak week                 (hidden)
    d  = rng.uniform(-1.5, -0.4)     # stringency additive effect
    e  = rng.uniform(0.8, 2.2)       # testing additive effect
    g  = rng.uniform(0.8, 1.6)       # baseline reporting offset
    return A, a, b, tp, d, e, g


def fval(x, cf):
    A, a, b, tp, d, e, g = cf
    u = x[0] - tp
    curve = A / (math.exp(-a * u) + math.exp(b * u))
    return curve + d * x[1] + e * x[2] + g


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.5 + (t - 1) * 0.30       # absolute measurement noise
    n = 220 - (t - 1) * 16             # 220 .. 76 reported weeks
    cf = coeffs(t)
    rng = random.Random(400 + t * 104729)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        x = [rng.uniform(T0, T1),
             rng.uniform(0.0, 1.0),
             rng.uniform(0.0, 1.0),
             rng.uniform(0.0, 1.0)]
        y = fval(x, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r %r %r" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

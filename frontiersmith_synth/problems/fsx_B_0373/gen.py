#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Artisan-bakery supply-chain bench.  Each production day a per-line hidden
"fresh-yield law" couples four normalised supply-chain readings

    x1 = flour-silo inventory level      x2 = proofing-room temperature
    x3 = forecast order volume           x4 = ingredient delivery lead-time
                                              (staleness of the incoming stock)

to the delivered next-day sellable-loaf yield index y (normalised).  Forecast
demand is ATTENUATED EXPONENTIALLY as delivery lead-time grows (stale stock
depresses the achievable bake), the flour level and proof temperature interact
multiplicatively (dough rise), proof temperature contributes a square-law
over/under-proofing curvature, plus a mild linear flour trend and an offset.
Each test id fixes a DIFFERENT hidden production line (different pump of
ingredients / proofing curve) with its own exponential staleness-decay rate --
the number the solver must recover.

The line manager only ever samples inside the SAFE operating core, x_i in
[0,1], and every logged reading carries scale-ripple + counting noise.  The
held-out grading split lives on the OVER-RANGE frontier (higher demand / longer
lead-time / hotter room) and is regenerated inside the grader only -- it is
never printed here.

Difficulty ladder (testId 1..10): more logging noise + fewer sampled days.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows.
The hidden law and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    rng = random.Random(618037 + t * 40961)
    a = rng.uniform(1.5, 3.0)     # x3*exp(-c*x4)   demand * staleness-decay envelope
    c = rng.uniform(0.60, 1.10)   # HIDDEN exponential staleness-decay rate
    b = rng.uniform(1.0, 2.0)     # x1*x2           flour * proof-temp interaction
    d = rng.uniform(1.0, 2.5)     # x2^2            proof-temp curvature (square law)
    e = rng.uniform(0.5, 1.5)     # x1              linear flour trend
    g = rng.uniform(-0.5, 0.5)    # offset
    return a, c, b, d, e, g


def fval(x, cf):
    a, c, b, d, e, g = cf
    return (a * x[2] * math.exp(-c * x[3]) + b * x[0] * x[1]
            + d * x[1] * x[1] + e * x[0] + g)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.06 + (t - 1) * 0.045
    n = 380 - (t - 1) * 20
    cf = coeffs(t)
    rng = random.Random(4409 + t * 71993)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]
        y = fval(x, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r %r %r" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

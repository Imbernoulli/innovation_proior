#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Archaeology dig grid: a hidden per-site "artifact density" law couples four
normalised survey readings taken at each grid cell of an excavation

    x1 = stratigraphic depth index      x2 = soil phosphate concentration
    x3 = magnetometer gradient          x4 = distance to buried watercourse

to the recovered artifact density y (finds per cubic metre, normalised).  Each
test id fixes a DIFFERENT hidden law (a different dig site with different
soil/geophysics).  The surveyor only ever samples the WELL-EXCAVATED core of
the trench, x_i in [0,1], and every reading carries instrument + counting
noise.  The held-out grading split lives in the deeper / farther UNEXCAVATED
frontier region and is regenerated inside the grader only -- it is never
printed here.

Difficulty ladder (testId 1..10): more measurement noise + fewer sampled cells.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows.
The hidden law and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    rng = random.Random(310007 + t * 6221)
    a = rng.uniform(1.5, 3.0)     # x1*x2  (depth x phosphate interaction)
    b = rng.uniform(2.0, 3.5)     # exp scale
    c = rng.uniform(0.50, 0.85)   # exp rate (HIDDEN magnetometer response)
    d = rng.uniform(1.0, 2.5)     # x4^2  (distance-to-water curvature)
    e = rng.uniform(0.5, 1.5)     # x1 linear
    g = rng.uniform(-0.5, 0.5)    # offset
    return a, b, c, d, e, g


def fval(x, cf):
    a, b, c, d, e, g = cf
    return a * x[0] * x[1] + b * math.exp(c * x[2]) + d * x[3] * x[3] + e * x[0] + g


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.07 + (t - 1) * 0.05
    n = 220 - (t - 1) * 16
    cf = coeffs(t)
    rng = random.Random(640 + t * 101771)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]
        y = fval(x, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r %r %r" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

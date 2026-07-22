#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE calibration sample to stdout.

Alchemist limiting-reagent brewing: a hidden per-recipe stoichiometric vector
s = (s1,s2,s3,s4) of SMALL COPRIME INTEGERS in 1..6, plus a hidden real gain
g, define the true yield law

    y = g * min(q1/s1, q2/s2, q3/s3, q4/s4)

Each test id fixes a DIFFERENT hidden recipe. The solver only ever sees noisy
CALIBRATION brews clustered near a BALANCED ratio (every reagent's q_i/s_i is
close to a common per-brew batch scale b, so no single reagent is drastically
limiting). The held-out grading split lives in an UNBALANCED extrapolation
regime (one reagent deliberately kept scarce) and is regenerated inside the
grader only -- it is never printed here.

Difficulty ladder (testId 1..10): more measurement noise + fewer samples.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
(q1 q2 q3 q4 y). The hidden recipe (s, g) and the RNG seed are NOT printed.
"""
import sys, math, random

K = 4


def coeffs(t):
    """Hidden recipe: small coprime stoichiometric vector + a positive gain."""
    rng = random.Random(90101 + t * 7919)
    s_max = 6
    while True:
        s = [rng.randint(1, s_max) for _ in range(K)]
        g0 = s[0]
        for v in s[1:]:
            g0 = math.gcd(g0, v)
        if g0 == 1:
            break
    g = rng.uniform(3.0, 7.0)
    return s, g


def gen_train(t):
    s, g = coeffs(t)
    n = 220 - (t - 1) * 16
    sigma = 0.04 + (t - 1) * 0.025          # relative measurement noise on y
    rng = random.Random(500 + t * 104729)
    rows = []
    for _ in range(n):
        b = rng.uniform(1.5, 4.5)           # common per-brew batch scale
        u = [rng.uniform(0.82, 1.18) for _ in range(K)]  # near-balanced jitter
        q = [s[i] * b * u[i] for i in range(K)]
        m = min(q[i] / s[i] for i in range(K))
        y_true = g * m
        y = y_true * (1.0 + rng.gauss(0.0, sigma))
        rows.append((q, y))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for q, y in rows:
        out.append("%r %r %r %r %r" % (q[0], q[1], q[2], q[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

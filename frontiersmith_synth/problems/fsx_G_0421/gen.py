#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE (compute, data, loss) training table to stdout.

LLM-pretraining scaling law.  Each test id fixes a DIFFERENT hidden neural
scaling law that maps

    C = training compute  (normalised units, ~ 1e18 FLOP each)
    D = training tokens   (normalised units, ~ 1e9 tokens each)

to the achieved validation cross-entropy loss `L`.  The true law has the
Chinchilla-style shape

    L(C, D) = E  +  A * C**(-alpha)  +  B * D**(-beta)

with an *irreducible entropy floor* E > 0 that no amount of compute can beat.
The solver only ever sees noisy measurements from a bank of SMALL exploratory
runs (small C, small D).  The held-out grading split lives at MUCH larger
compute/data (a genuine extrapolation to bigger scale) and is regenerated
inside the grader only -- it is never printed here.

Difficulty ladder (testId 1..10): more measurement noise + fewer runs.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
"C D L".  The hidden law, its coefficients and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    """Hidden scaling-law coefficients for test id t (lives here and in verify.py)."""
    rng = random.Random(31337 + t * 6997)
    E = rng.uniform(1.55, 1.95)      # irreducible entropy floor
    A = rng.uniform(6.0, 14.0)       # compute prefactor
    alpha = rng.uniform(0.22, 0.40)  # compute exponent (HIDDEN)
    B = rng.uniform(4.0, 10.0)       # data prefactor
    beta = rng.uniform(0.22, 0.40)   # data exponent (HIDDEN)
    return E, A, alpha, B, beta


def fval(C, D, cf):
    E, A, alpha, B, beta = cf
    return E + A * C ** (-alpha) + B * D ** (-beta)


def loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.020 + (t - 1) * 0.010     # measurement noise on the loss
    n = 120 - (t - 1) * 8               # 120 -> 48 exploratory runs
    cf = coeffs(t)
    rng = random.Random(8000 + t * 100003)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        # small exploratory-run region
        C = loguniform(rng, 1.0, 80.0)
        D = loguniform(rng, 1.0, 50.0)
        L = fval(C, D, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r" % (C, D, L))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

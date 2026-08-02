#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Queue-saturation forecasting.  A hidden single-server queue has a capacity
(offered-load asymptote) C: mean wait time blows up hyperbolically as the
offered load L approaches C.  Wait time is also scaled by the arrival
burstiness covariate B (a squared-coefficient-of-variation-style measure of
how clumped arrivals are; B=1 ~ Poisson arrivals, B>1 ~ bursty, B<1 ~ regular).
Each testId fixes a DIFFERENT hidden queue (capacity, sensitivity, burstiness
exponent).

The solver only ever SEES this TRAIN sample, which is logged while the
system was run under LOW-to-MODERATE offered load (never above roughly half
of capacity).  In that sub-saturation range wait time looks almost linear in
L -- the hyperbolic term contributes only a MILD curvature.  The held-out
grading load range is regenerated only inside the grader, at HIGHER offered
load the solver never observed -- it is never printed here.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"<load> <burstiness> <mean_wait>".  The hidden capacity, sensitivity,
burstiness exponent and RNG seeds are NOT printed anywhere.
"""
import sys, random, math


def hidden_params(t):
    """Hidden queue for this test id (lives in gen AND grader, never printed).
    C          -- capacity (offered-load asymptote); wait -> infinity as L -> C
    alpha      -- base wait sensitivity (Poisson-ish component)
    beta       -- burstiness sensitivity
    p          -- true burstiness exponent (near, but not exactly, the textbook
                  Kingman value of 2 -- real systems drift a little; a solver
                  that hard-codes exponent 2 leaves a small residual)
    """
    rng = random.Random(700003 + t * 91711)
    C = rng.uniform(50.0, 200.0)
    alpha = rng.uniform(0.5, 1.1)
    beta = rng.uniform(0.5, 1.4)
    p = rng.uniform(1.7, 2.3)
    return C, alpha, beta, p


def schedule(t):
    """Difficulty ladder: visible-range ceiling shrinks and noise grows with t,
    so the mild curvature that reveals C gets progressively harder to see."""
    u_train_max = [0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36][t - 1]
    sigma_mult  = [0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04][t - 1]
    n_train     = [70, 70, 65, 65, 60, 60, 55, 55, 50, 50][t - 1]
    return u_train_max, sigma_mult, n_train


def true_wait(L, B, C, alpha, beta, p):
    """Kingman-style wait time: linear response x utilization-blowup factor."""
    return (alpha + beta * (B ** p)) * L / (C - L)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    C, alpha, beta, p = hidden_params(t)
    u_train_max, sigma_mult, n_train = schedule(t)
    rng = random.Random(1000003 + t * 7919)

    rows = []
    for _ in range(n_train):
        u = rng.uniform(0.03, u_train_max)      # sub-saturation offered load only
        L = u * C
        B = rng.uniform(0.2, 2.2)
        w = true_wait(L, B, C, alpha, beta, p)
        noise = rng.gauss(0.0, sigma_mult)       # multiplicative log-normal-ish jitter
        w_obs = w * math.exp(noise)
        rows.append((L, B, w_obs))

    out = ["%d %d" % (n_train, t)]
    for L, B, w in rows:
        out.append("%.6f %.6f %.6f" % (L, B, w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

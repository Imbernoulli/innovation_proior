#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Coffee percolating through sieves of many sizes.  A sieve of mesh-count L is a
finite lattice; a batch of coffee grounds is poured at packing density p, and
we record the fraction of pours (out of many seeded trials) that find a path
clear through the sieve: Pi_hat(p, L).  Even a fully-packed sieve occasionally
clumps and blocks every path (so the reading never quite reaches 1), and even
a nearly-empty sieve sometimes lets a trace of coffee through by capillary
action (so it never quite reads 0) -- both are folded into the fixed constants
OFF, AMP below, which are NOT the tunable part of the law.

The hidden truth (never printed) is a finite-size-scaling crossing law

    Pi_true(p, L) = OFF + AMP * sigmoid( (p - pc) * L**(1/nu) )

for an unknown critical density pc and an unknown exponent nu that is SHARED
by BOTH the threshold location's drift with L and the transition's sharpening
with L -- one exponent governs both.  A DIFFERENT (pc, nu) is drawn per test
id.  The solver only ever sees SMALL-sieve data (L in {8,16,24,32}); grading
happens on much larger sieves (see verify.py) -- a genuine extrapolation in L.

STDOUT prints ONLY: a header "<n_rows> <test_id>" then n_rows rows
"<L> <p> <Pi_hat>".  The hidden pc, nu, and RNG seed are NEVER printed.
"""
import sys, math, random

OFF, AMP = 0.1, 0.8
TRAIN_LS = [8, 16, 24, 32]
P_GRID = [round(0.20 + 0.05 * i, 2) for i in range(13)]  # 0.20 .. 0.80 step 0.05


def hidden_params(t):
    """Hidden crossing law for this test id (also re-derived in verify.py)."""
    rng = random.Random(9013 + t * 7919)
    pc = rng.uniform(0.30, 0.70)
    nu = rng.uniform(0.9, 2.2)
    return pc, nu


def _sig(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def pi_true(p, L, pc, nu):
    x = (p - pc) * (L ** (1.0 / nu))
    return OFF + AMP * _sig(x)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pc, nu = hidden_params(t)
    # difficulty ladder: noisier readings and a sparser p-grid at higher t
    sigma = 0.02 + 0.006 * (t - 1)
    stride = 1 + (t - 1) // 4          # 1 for t<=4, 2 for t in 5..8, 3 for t in 9..10
    p_grid = P_GRID[::stride]
    if p_grid[-1] != P_GRID[-1]:
        p_grid = p_grid + [P_GRID[-1]]

    rng = random.Random(551977 + t * 104729)
    rows = []
    for L in TRAIN_LS:
        for p in p_grid:
            true_pi = pi_true(p, L, pc, nu)
            obs = true_pi + rng.gauss(0.0, sigma)
            obs = min(1.0, max(0.0, obs))
            rows.append((L, p, obs))

    out = ["%d %d" % (len(rows), t)]
    for L, p, pih in rows:
        out.append("%d %.6f %.6f" % (L, p, pih))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

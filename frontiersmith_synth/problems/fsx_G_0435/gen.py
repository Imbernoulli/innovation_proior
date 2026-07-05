#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Options desk / implied-volatility surface recovery.  A hidden, deterministic
implied-vol functional sigma(k, t) governs an equity index option book, where
    k = log-moneyness  = log(K / F)      (K = strike, F = forward)
    t = tenor          in years
The desk only trades a LIQUID window of NEAR-THE-MONEY strikes during the
session, so training quotes are sampled from a narrow moneyness band
    k in [-0.15, 0.15],  t in [0.10, 2.00]
and every quote carries a small bid/ask microstructure error.  The grader's
HELD-OUT split lives in the FAR-STRIKE WINGS (deep OTM / ITM, |k| well beyond
the traded band) and is regenerated inside the grader only -- it is never
printed here.  The wing convexity ("smile") that dominates there must be
inferred from the near-ATM data, not memorised.

Each test id fixes a DIFFERENT hidden surface (a different book).
Difficulty ladder (testId 1..10): more quote noise + fewer training strikes.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
"k t sigma".  The hidden law and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    # Hidden per-book surface parameters (never printed).
    rng = random.Random(60413 + t * 7919)
    a = rng.uniform(0.15, 0.26)      # base ATM vol level
    term = rng.uniform(-0.03, 0.05)  # term-structure slope in t
    skew = rng.uniform(-0.55, -0.20) # equity skew (linear in k)
    smile = rng.uniform(0.60, 1.20)  # base convexity (k^2)
    conv = rng.uniform(0.50, 1.50)   # extra short-dated convexity
    rate = rng.uniform(0.80, 1.80)   # smile-decay rate in t (HIDDEN, per book)
    return a, term, skew, smile, conv, rate


def fval(k, t, cf):
    a, term, skew, smile, conv, rate = cf
    # ATM level + term structure + linear skew + smile convexity that
    # flattens with maturity (short-dated smiles are steeper).
    return (a + term * t + skew * k + smile * k * k
            + conv * k * k * math.exp(-rate * t))


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma_noise = 0.0040 + (t - 1) * 0.0016
    n = 200 - (t - 1) * 14
    cf = coeffs(t)
    rng = random.Random(311 + t * 104729)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        k = rng.uniform(-0.15, 0.15)
        tt = rng.uniform(0.10, 2.00)
        s = fval(k, tt, cf) + rng.gauss(0.0, sigma_noise)
        out.append("%r %r %r" % (k, tt, s))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

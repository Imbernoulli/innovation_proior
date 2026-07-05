#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE in-band training sample to stdout.

Antenna frequency-response modelling.  A resonant receiving antenna behaves,
around its fundamental resonance, like a second-order band-pass network: its
measured POWER response

    P(f) = K * f^2 / ( (f0^2 - f^2)^2 + (f0*f/Q)^2 )

is a rational transfer function of the (normalised) frequency f, with a hidden
resonant frequency f0, quality factor Q and gain K.  A network analyser only
sweeps the IN-BAND region around resonance; the solver receives those noisy
in-band samples.  The grading split is a set of OUT-OF-BAND frequencies (the
low- and high-frequency roll-off skirts) and is regenerated inside the grader
only -- it is never printed here.

Each test id fixes a DIFFERENT hidden antenna (f0, Q, K).  Difficulty ladder
(testId 1..10): more measurement noise + fewer swept points.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows
"<f> <P>".  The hidden law, its parameters and the seed are NOT printed.
"""
import sys, random


def coeffs(t):
    rng = random.Random(60413 + t * 7919)
    f0 = rng.uniform(0.80, 1.30)     # resonant frequency (HIDDEN)
    Q = rng.uniform(2.50, 6.00)      # quality factor    (HIDDEN)
    K = rng.uniform(0.80, 2.50)      # gain              (HIDDEN)
    return f0, Q, K


def presp(f, cf):
    f0, Q, K = cf
    den = (f0 * f0 - f * f) ** 2 + (f0 * f / Q) ** 2
    return K * f * f / den


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cf = coeffs(t)
    f0, Q, K = cf
    peak = K * Q * Q / (f0 * f0)                 # response at resonance
    sigma = (0.03 + (t - 1) * 0.02) * peak       # additive analyser noise
    n = 120 - (t - 1) * 8                        # 120 down to 48
    rng = random.Random(500 + t * 104729)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        f = rng.uniform(0.15, 2.20)              # IN-BAND sweep
        y = presp(f, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r" % (f, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

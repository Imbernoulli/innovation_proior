#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE growth-chamber logbook (training sample) to stdout.

Theme: predict growth beyond what you measured.  A simulated organism is grown
under a fixed control parameter p (a nutrient index, 0<p<1) for several
generations of the same chamber.  At each integer time step t the chamber logs
three descriptors of the organism: its size S, its aspect ratio AR, and its
branch-tip count TC.  Size follows a fixed but UNKNOWN two-part law: a
carrying-capacity amplitude that scales with p as a power law (an allometric
scaling relation) and an approach RATE that scales with (1-p) as a different
power law -- both amplitude and rate saturate the growth curve toward a
capacity, never printed.

Crucially every logbook only ever runs the chamber at p INSIDE a narrow
interior band (p in [0.42, 0.58]).  The grader later asks for a prediction of
size at EXTREME p (near 0.05 or near 0.95) -- a genuinely different regime that
is regenerated only inside the grader and never printed here.

Difficulty ladder (testId 1..10): the true power-law exponents move further
from 1 (stronger curvature -> a bigger extrapolation trap) and the measurement
noise grows, as testId increases.

STDOUT prints ONLY: a header "<n_rows> <test_id>" then n_rows rows
    t p S AR TC
The exponents, amplitudes, rates and the seed are NOT printed.
"""
import sys
import math
import random

TRAIN_PS = [0.42, 0.46, 0.50, 0.54, 0.58]
TMAX = 10


def params(tid):
    """Hidden per-test constants (identical formulas duplicated in verify.py).
    alpha, beta are the two allometric scaling exponents (kept well away from
    1 and shrinking with tid so the power-law curvature -- invisible in the
    narrow interior band -- gets sharper and sharper toward the extremes).
    K0, r0 (the overall size scale) are held FIXED across the ladder so the
    difficulty ladder isolates curvature strength + measurement noise, not a
    confounded change in absolute size."""
    alpha = max(0.12, 0.90 - 0.075 * tid)
    beta = max(0.12, 0.85 - 0.065 * tid)
    K0 = 10.0
    r0 = 0.15
    sigma = 0.10 + 0.03 * tid
    return alpha, beta, K0, r0, sigma


def true_S(t, p, alpha, beta, K0, r0):
    """Saturating growth law: capacity K(p)=K0*p^alpha (allometric in p),
    rate r(p)=r0*(1-p)^beta (allometric in 1-p); S(t,p)=K(p)*(1-exp(-r(p)*t))."""
    K = K0 * (p ** alpha)
    r = r0 * ((1.0 - p) ** beta)
    return K * (1.0 - math.exp(-r * t))


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    alpha, beta, K0, r0, sigma = params(tid)
    rng = random.Random(802103 + tid * 61717)

    rows = []
    for p in TRAIN_PS:
        for t in range(1, TMAX + 1):
            s_true = true_S(t, p, alpha, beta, K0, r0)
            s_obs = s_true + rng.gauss(0.0, sigma)
            base = math.sqrt(max(s_true, 0.0))
            ar_true = 1.15 + 0.55 * base / (1.0 + 0.35 * base)
            ar_obs = ar_true + rng.gauss(0.0, sigma * 0.02)
            tc_true = 2.0 + 3.4 * (max(s_true, 0.0) ** 0.65)
            tc_obs = max(0, round(tc_true + rng.gauss(0.0, 0.4)))
            rows.append((t, p, s_obs, ar_obs, tc_obs))

    print("%d %d" % (len(rows), tid))
    out = []
    for t, p, s, ar, tc in rows:
        out.append("%d %.6f %.6f %.6f %d" % (t, p, s, ar, tc))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

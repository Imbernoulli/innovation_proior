#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE TRAIN sample of censored call-center hold
times to stdout.

Physical picture (call-center hold times with hang-ups): callers arrive when
the center is running at load `rho` in (0, rho_max), where `rho_max` is the
(unknown, unpublished) load at which the center's staffing can no longer keep
up and mean hold time DIVERGES. Below that ceiling the mean hold time follows
a near-saturation law

    mean_hold(rho) = C / (1 - rho / rho_max)

Every individual call's hold time is then drawn as an exponential deviate
around that mean (queueing systems are famously high-variance, not just
noisy-around-a-line). BUT the call center enforces a hard patience cap `T`:
any caller who would have waited longer than `T` hangs up, and the logger
records that call's hold time as exactly `T` (right-censoring). Training data
is only ever collected while the center runs at COMFORTABLE loads (well
below rho_max) -- grading happens at NEAR-CRITICAL loads the training data
never reaches.

The hidden constants (C, rho_max) and the cap T's numeric consequences are
never printed as such -- only sampled (rho, hold) pairs, plus the
publicly-known cap value T itself (a caller can always tell you what the
posted hang-up policy is). Held-out grading lives only inside the checker.

STDOUT: a header "<n_train> <test_id> <T>" then n_train lines "rho hold".
"""
import sys, math, random

RHO_MAX = 0.92
C = 0.55
T_CAP = 3.0

TRAIN_LOW_FRAC = 0.03   # rho in [TRAIN_LOW_FRAC, TRAIN_HIGH_FRAC] * RHO_MAX
TRAIN_HIGH_FRAC = 0.80


def true_mean_hold(rho, rho_max=RHO_MAX, c=C):
    return c / (1.0 - rho / rho_max)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    n_train = 500 - 15 * (t - 1)   # 500 .. 365: fewer samples -> noisier bins as t grows

    rnd = random.Random(654321 + 233 * t)
    lo = TRAIN_LOW_FRAC * RHO_MAX
    hi = TRAIN_HIGH_FRAC * RHO_MAX

    rows = []
    for _ in range(n_train):
        rho = rnd.uniform(lo, hi)
        m = true_mean_hold(rho)
        u = rnd.random()
        if u < 1e-15:
            u = 1e-15
        raw = -m * math.log(u)   # exponential deviate with mean m
        hold = raw if raw < T_CAP else T_CAP
        rows.append((rho, hold))

    out = ["%d %d %.6f" % (n_train, t, T_CAP)]
    for rho, hold in rows:
        out.append("%.6f %.6f" % (rho, hold))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

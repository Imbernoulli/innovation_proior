#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE staggered-cohort novelty-decay notebook to stdout.

Setting: a product feature is rolled out to COHORTS of users that ENTER treatment
on different calendar days (staggered adoption).  Every day each currently-active
cohort's treatment lift is measured.  A cohort's true lift is highest right after
it enters (novelty / primacy) and decays toward a smaller PERSISTENT long-run
lift as the cohort ages.  On top of that, every cohort is ALSO nudged, on any
given calendar day, by a common company-wide wobble (traffic mix, a promo, a
holiday) that has NOTHING to do with how old the cohort is.

STDOUT prints ONLY: header "<n_rows> <test_id>" then n_rows lines
"<cohort> <entry_day> <calendar_day> <age> <lift>" (age = calendar_day - entry_day).
The hidden law, its coefficients, and the seeds are NEVER printed -- only rows.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
C_COHORTS = 6          # number of staggered cohorts visible in training
GAP       = 6          # days between successive cohorts' entry
W         = 16         # each cohort is observed for W consecutive ages: 0..W-1
T_VIS     = (C_COHORTS - 1) * GAP + W     # last calendar day appearing in training
PERIOD1   = T_VIS / 3.0
PERIOD2   = T_VIS / 5.0


def params(t):
    """Hidden novelty-decay law for this test id (identical in gen.py / verify.py)."""
    rng = random.Random(521000 + t * 8123911)
    P = rng.uniform(0.04, 0.10)                              # persistent long-run lift
    ratio_lo, ratio_hi = 0.8 + 0.15 * t, 1.6 + 0.35 * t       # trap ramps up with test id
    A = P * rng.uniform(ratio_lo, ratio_hi)                   # novelty/primacy amplitude
    tau_lo, tau_hi = 4.0 + 0.3 * t, 10.0 + 1.2 * t
    tau = rng.uniform(tau_lo, tau_hi)                         # decay time constant (days)
    D_amp = P * rng.uniform(0.3, 0.9)                         # common calendar wobble scale
    phase1 = rng.uniform(0.0, 2 * math.pi)
    phase2 = rng.uniform(0.0, 2 * math.pi)
    sigma_train = rng.uniform(0.004, 0.010)
    sigma_held = sigma_train * rng.uniform(1.2, 1.6)
    return dict(P=P, A=A, tau=tau, D_amp=D_amp, phase1=phase1, phase2=phase2,
                sigma_train=sigma_train, sigma_held=sigma_held)


def calendar_wobble(t_cal, prm):
    """Company-wide daily wobble: identical for every cohort measured on t_cal, has
    nothing to do with any cohort's age.  Same on any calendar day regardless of who
    is observed that day -- this is what staggered cohorts let you cancel out."""
    d = (0.6 * prm["D_amp"] * math.sin(2 * math.pi * t_cal / PERIOD1 + prm["phase1"])
         + 0.4 * prm["D_amp"] * math.sin(2 * math.pi * t_cal / PERIOD2 + prm["phase2"]))
    return d


def true_lift(age, t_cal, prm):
    return prm["P"] + prm["A"] * math.exp(-age / prm["tau"]) + calendar_wobble(t_cal, prm)


def gen_train(t):
    prm = params(t)
    rng = random.Random(60013 + t * 977)
    rows = []
    for c in range(1, C_COHORTS + 1):
        s_c = 1 + (c - 1) * GAP
        for age in range(W):
            t_cal = s_c + age
            L = true_lift(age, t_cal, prm) + rng.gauss(0.0, prm["sigma_train"])
            rows.append((c, s_c, t_cal, age, L))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for c, s_c, t_cal, age, L in rows:
        out.append("%d %d %d %d %.8g" % (c, s_c, t_cal, age, L))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

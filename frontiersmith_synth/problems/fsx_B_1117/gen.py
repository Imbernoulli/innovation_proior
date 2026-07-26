#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Hidden reaction: a substrate S is converted by a fixed amount of catalyst C.
The TRUE per-catalyst rate law is a saturating (Michaelis-Menten-like) form

    rate(S, C) = Vmax * C * S / (Km + S)

but the solver is only ever shown assay points recorded in the DILUTE regime
(S well below Km, across a few different catalyst loadings C). In that
regime rate(S,C) ~= (Vmax/Km) * C * S -- i.e. it LOOKS first-order (linear)
in S. The held-out grading assay (regenerated only inside the checker) probes
S values well beyond what was trained on, where the true law saturates.

STDOUT prints ONLY a header "<test_id> <n_regimes> <n_pts>" followed by
n_regimes*n_pts rows "S C rate". The hidden Vmax, Km, seed and catalyst list
are NEVER printed.
"""
import sys, random


def seed_params(test_id):
    """Hidden reaction constants for this test id (duplicated verbatim in verify.py)."""
    rng = random.Random(900001 + test_id * 7919)
    if test_id <= 3:
        n_regimes, n_pts = 2, 6
    elif test_id <= 7:
        n_regimes, n_pts = 3, 8
    else:
        n_regimes, n_pts = 4, 9
    Vmax = rng.uniform(20.0, 70.0)
    Km = rng.uniform(3.0, 14.0)
    noise_frac = rng.uniform(0.03, 0.06)        # TRAIN assay measurement noise
    noise_frac_ho = rng.uniform(0.16, 0.26)     # HELD-OUT assay measurement noise
    pool = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    rng.shuffle(pool)
    C_values = sorted(pool[:n_regimes])
    dilute_frac = rng.uniform(0.28, 0.55)       # train S reaches this fraction of Km
    return dict(Vmax=Vmax, Km=Km, noise_frac=noise_frac, noise_frac_ho=noise_frac_ho,
                C_values=C_values, n_pts=n_pts, dilute_frac=dilute_frac, pool=pool,
                n_regimes=n_regimes)


def rate_true(S, C, Vmax, Km):
    return Vmax * C * S / (Km + S)


def make_train(test_id, p):
    """The dilute-regime TRAIN assay (identical code lives in verify.py so the
    checker's own constant baseline reproduces exactly what the solver sees)."""
    Vmax, Km = p["Vmax"], p["Km"]
    C_values = p["C_values"]
    n_pts = p["n_pts"]
    dilute_hi = p["dilute_frac"] * Km
    rng = random.Random(31337 + test_id * 13)
    rows = []
    for C in C_values:
        for k in range(n_pts):
            frac = (k + 1) / (n_pts + 1)
            S = 0.04 * Km + frac * (dilute_hi - 0.04 * Km)
            S *= rng.uniform(0.97, 1.03)
            tr = rate_true(S, C, Vmax, Km)
            noisy = tr + rng.gauss(0.0, p["noise_frac"] * tr)
            noisy = max(0.0, noisy)
            rows.append((S, C, noisy))
    return rows


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = seed_params(test_id)
    rows = make_train(test_id, p)
    out = ["%d %d %d" % (test_id, p["n_regimes"], p["n_pts"])]
    for S, C, r in rows:
        out.append("%.6f %.6f %.6f" % (S, C, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

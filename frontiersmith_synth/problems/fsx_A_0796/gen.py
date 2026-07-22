#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy "relay log" to stdout.

Backstory (for gen.py's own bookkeeping only -- NOT printed): a lighthouse's
mechanical flash-counter relay counts passing ships. Every ship advances an
internal 3-position calibration gear by one notch (the gear position
"s = n mod 3" is a genuine finite-state transducer state: it depends only on
how many ships have been counted, cycling deterministically). Whichever gear
is currently engaged applies ITS OWN linear-fractional (rational) recalibration
map to the running ship count n to produce the register reading:

    y(n) = (a_s * n + b_s) / (c_s * n + d_s)          s = n mod 3

Each of the 3 gears has its own (a_s, b_s, c_s, d_s); all four are positive
integers so the map is monotone and the register reading settles toward a
finite asymptote a_s / c_s as n grows, but is NOT close to that asymptote
over the modest ship counts the historical logbook covers -- the curvature is
still very much in play.

The TRAIN log the solver sees comes from the historical logbook: ship counts
n in [4, 90] (a slow harbor season), with small measurement noise. STDOUT
prints ONLY: header "<n_rows> <test_id>" then n_rows data rows "n s y". The
gear formulas, their coefficients and the seed are never printed -- only
these three numeric columns.
"""
import sys, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
K = 3
N_TRAIN = 90
N_TRAIN_LO, N_TRAIN_HI = 4, 90
NOISE_TRAIN = 0.04


def gear_params(t):
    """Hidden per-gear rational coefficients for this test id (identical in
    gen.py and verify.py)."""
    rng = random.Random(700000 + t * 9176321)
    ps = []
    for _s in range(K):
        a = rng.randint(3, 9)
        b = rng.randint(0, 20)
        c = rng.randint(1, 4)
        d = rng.randint(100, 600)
        ps.append((a, b, c, d))
    return ps


def true_y(n, s, ps):
    a, b, c, d = ps[s]
    return (a * n + b) / (c * n + d)


def gen_rows(t, n_rows, n_lo, n_hi, noise_sigma, seed_base):
    ps = gear_params(t)
    rng = random.Random(seed_base + t * 131)
    rows = []
    for _ in range(n_rows):
        n = rng.randint(n_lo, n_hi)
        s = n % K
        y = true_y(n, s, ps) * (1.0 + rng.gauss(0.0, noise_sigma))
        rows.append((n, s, y))
    return rows


def gen_train(t):
    return gen_rows(t, N_TRAIN, N_TRAIN_LO, N_TRAIN_HI, NOISE_TRAIN, 111)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for n, s, y in rows:
        out.append("%d %d %.8g" % (n, s, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

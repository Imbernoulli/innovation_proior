#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Sheet-metal bending: a die of radius r bends a sheet of thickness t; after the
punch releases, the sheet "springs back" by an amount S. Each testId is a
DIFFERENT alloy/tempering (hidden law). The physical reality (never printed,
lives only in gen.py + verify.py) is that TWO distinct mechanisms govern S as
a function of x = r/t:

  * for x below a critical ratio, the bend is elastic-dominated: S grows
    roughly LINEARLY in x.
  * for x above the critical ratio, the bend is plastic-saturated: S grows
    much more SLOWLY (a fractional power of x, plus an offset).

The critical ratio is itself NOT a fixed number -- it drifts with sheet
thickness t. The TRAIN sample here is drawn from THIN sheets only, where the
critical ratio is large, so MOST (not all) sampled (r,t) pairs are
elastic-dominated. The HELD-OUT grading sample (regenerated only inside
verify.py, never printed here) is drawn from THICKER sheets, where the
critical ratio has shifted DOWN -- so the held-out sample is a genuine mix of
both regimes, with plastic-dominated pairs now the majority. Same r/t values
can therefore sit in OPPOSITE regimes depending on t: a model that never
learns t's role in moving the boundary gets a systematic share of EITHER
split wrong, whichever regime it defaults to.

STDOUT prints ONLY: a header "<n_train> <case_id>" then n_train rows of
"<r> <t> <S>". Hidden coefficients / seed are NOT printed.
"""
import sys, random

BASE = 918273
T_TRAIN = (0.5, 1.0)
# NOTE: verify.py's held-out T_HOLD = (1.4, 2.2) must keep BOTH regimes present
# in the held-out sample (not saturate to ~100% one branch) -- see verify.py.
X_RANGE = (2.0, 30.0)   # r/t ratio range -- SAME domain used by verify.py's held-out split
XC_FLOOR = 0.5


def hidden_params(cid):
    """Hidden alloy law for this test id (also reconstructed in verify.py)."""
    rng = random.Random(BASE + cid * 104729)
    A = 30.0 + rng.uniform(-2.0, 2.0)     # boundary intercept
    Bc = 11.0 + rng.uniform(-1.5, 1.5)    # boundary slope in t
    c1 = rng.uniform(0.055, 0.095)        # elastic-branch slope
    c2 = rng.uniform(0.85, 1.25)          # plastic-branch coefficient
    c3 = rng.uniform(-0.12, 0.12)         # plastic-branch offset
    return A, Bc, c1, c2, c3


def xc(A, Bc, t):
    return max(XC_FLOOR, A - Bc * t)


def true_S(r, t, A, Bc, c1, c2, c3):
    x = r / t
    if x < xc(A, Bc, t):
        return c1 * x
    return c2 * (x ** (1.0 / 3.0)) + c3


def main():
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    A, Bc, c1, c2, c3 = hidden_params(cid)
    sigma = 0.40 + 0.050 * (cid - 1)
    n = 260 - 6 * (cid - 1)

    rng_s = random.Random(55555 + cid * 7919)     # sampling stream (r, t)
    rng_n = random.Random(99991 + cid * 104729)   # noise stream

    lines = ["%d %d" % (n, cid)]
    for _ in range(n):
        t = rng_s.uniform(*T_TRAIN)
        x = rng_s.uniform(*X_RANGE)
        r = x * t
        s_true = true_S(r, t, A, Bc, c1, c2, c3)
        s_obs = s_true + rng_n.gauss(0.0, sigma)
        lines.append("%.6f %.6f %.6f" % (r, t, s_obs))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

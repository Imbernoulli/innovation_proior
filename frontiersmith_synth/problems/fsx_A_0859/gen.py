#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN log to stdout.

Jeweler's wobbling scale.  A balance scale is perfectly calibrated: the mean
reading at any load x is a fixed known constant mu.  But the WOBBLE (variance)
of repeated readings at a fixed load x is governed by a hidden two-regime law:

  x <  x0 :  sigma^2(x) = A * x^p                     (elastic spring: multiplicative)
  x >= x0 :  sigma^2(x) = A * x0^p + C * (x - x0)      (worn bearing: additive, continuous
                                                          at the knee x0)

Each testId fixes a DIFFERENT hidden (x0, p, A, C).  The solver only ever SEES
readings recorded over a LIGHT-load range [XLO, XHI]; the knee x0 sits inside
that range so both regimes are visible in training, but the grading readings
live at HEAVIER loads strictly beyond XHI (genuine extrapolation), regenerated
only inside the checker.

STDOUT prints ONLY: a header "<n_x> <R> <mu> <test_id>" then n_x rows of
"<x> <y_1> ... <y_R>" (R repeated readings at that load).  The hidden regime
law, its parameters, and the seed are NEVER printed.
"""
import sys, random, math

XLO, XHI = 0.3, 5.0
MU = 50.0
R = 14


def hidden_params(t):
    """Hidden wobble law for this test id (duplicated verbatim in verify.py)."""
    rng = random.Random(3010301 + t * 92821)
    # knee sits LATE in the training window: most training loads fall in
    # the near-flat regime, only a minority already show the ramp. A single
    # global fit dominated by the many flat points systematically undershoots
    # the ramp once extrapolated -- the intended trap.
    x0 = rng.uniform(2.6, 3.9)
    # regime 1 is deliberately near-FLAT (small p): the elastic spring's
    # slack absorbs load with almost no extra wobble. regime 2 is a much
    # STEEPER ramp: a single global power-law fit dragged flat by the many
    # regime-1 points badly underestimates the ramp once extrapolated.
    p = rng.uniform(0.05, 0.35)
    A = rng.uniform(0.5, 1.2)
    C = rng.uniform(1.0, 2.2)
    return x0, p, A, C


def wobble(x, x0, p, A, C):
    if x < x0:
        return A * (x ** p)
    return A * (x0 ** p) + C * (x - x0)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    x0, p, A, C = hidden_params(t)

    n_x = 46 - 2 * (t - 1)          # 46 .. 28: sparser training as t grows
    rng = random.Random(710501 + t * 1299709)

    rows = ["%d %d %.6f %d" % (n_x, R, MU, t)]
    for _ in range(n_x):
        x = rng.uniform(XLO, XHI)
        var = wobble(x, x0, p, A, C)
        sd = math.sqrt(max(1e-12, var))
        ys = [MU + rng.gauss(0.0, sd) for _ in range(R)]
        rows.append(("%.6f " % x) + " ".join("%.6f" % y for y in ys))

    sys.stdout.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

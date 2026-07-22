#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

A staged production line's output metric y responds to a control setting x
(the feed rate). The hidden TRUE law is a REGIME-SWITCHING process:

  x <  xc :  y = a0 + a1*x + a2*x^2                       (calm regime,
                                                             smooth, bounded
                                                             curvature)
  x >= xc :  y = y(xc) + B * (x - xc)^alpha                (congestion-
                                                             cascade regime:
                                                             once the feed
                                                             rate crosses the
                                                             hidden threshold
                                                             xc, backpressure
                                                             cascades through
                                                             the downstream
                                                             stages and the
                                                             excess output
                                                             grows as a power
                                                             law of the
                                                             DISTANCE PAST
                                                             THRESHOLD, with a
                                                             hidden scaling
                                                             exponent alpha)

Each testId fixes a different hidden (xc, a0, a1, a2, alpha, B). The training
band straddles xc (it runs from a fixed low feed rate up to a MODEST excess
past xc), so the solver sees SOME of both regimes -- but the held-out grading
grid (regenerated only inside the checker) reaches far deeper into the
congestion-cascade regime than any training row, testing genuine
extrapolation. Neither xc, a0, a1, a2, alpha, B nor the RNG seed are ever
printed here -- only noisy (x, y) rows.

STDOUT prints ONLY: a header "<testId> <N>" then N rows "x y".
"""
import sys, random, math

SEED_BASE = 882000
X_LO = 5.0
BAND_HI_EXTRA = 15.0     # training band reaches xc + BAND_HI_EXTRA
N_TRAIN = 90
NOISE_SIGMA = 0.025      # small multiplicative log-noise (measurement floor)

_PLAN_ALPHA = {1: 1.35, 2: 1.55, 3: 3.55, 4: 3.25, 5: 2.05,
               6: 1.45, 7: 3.75, 8: 2.55, 9: 3.05, 10: 1.85}


def hidden_law(t):
    """Hidden regime-switching law for this test id. Lives in gen AND
    checker; never printed."""
    rng = random.Random(SEED_BASE + t * 7919)
    xc = rng.uniform(12.0, 30.0)
    a0 = rng.uniform(1.0, 3.0)
    a1 = rng.uniform(0.15, 0.45)
    a2 = rng.uniform(0.004, 0.018)
    alpha = _PLAN_ALPHA.get(t, 2.20) + rng.uniform(-0.05, 0.05)
    B = rng.uniform(0.6, 2.2)
    return xc, a0, a1, a2, alpha, B


def y_true(x, xc, a0, a1, a2, alpha, B):
    if x < xc:
        return a0 + a1 * x + a2 * x * x
    y0 = a0 + a1 * xc + a2 * xc * xc
    return y0 + B * ((x - xc) ** alpha)


def train_rows(t):
    xc, a0, a1, a2, alpha, B = hidden_law(t)
    hi = xc + BAND_HI_EXTRA
    rng = random.Random(40410 + t * 13)
    rows = []
    for i in range(N_TRAIN):
        frac = (i + rng.uniform(0.05, 0.95)) / N_TRAIN
        frac = min(0.999999, max(0.000001, frac))
        x = X_LO + frac * (hi - X_LO)
        clean = y_true(x, xc, a0, a1, a2, alpha, B)
        noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((x, noisy))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = train_rows(t)
    out = ["%d %d" % (t, len(rows))]
    for x, y in rows:
        out.append("%.8f %.8f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE calibration log for the climate-sensor
piecewise-law discovery problem (format E).

A field station logs a scalar sensor response y (e.g. a melt / heat-flux index)
against four normalized drivers:
    x0 = temperature anomaly       (the regime-controlling driver)
    x1 = surface humidity index
    x2 = incident-solar index
    x3 = wind-shear index

The physical response is PIECEWISE: below a hidden temperature threshold the
sensor tracks a mild (roughly linear) law; once the anomaly crosses the threshold
an accelerating feedback ("tipping" term) switches on. The threshold, the
functional form, the coefficients, the sampling seed and the held-out
extrapolation region are NEVER printed here -- they live only inside the grader.

Difficulty ladder (testId 1..N): larger testId => FEWER training rows and HIGHER
measurement noise, and a smaller slice of data past the break, so the regime
change is harder to detect and extrapolate.

STDOUT is DATA ROWS ONLY: five whitespace-separated floats "x0 x1 x2 x3 y" per
line.
"""
import sys, math, random


def hidden_law(x0, x1, x2, x3):
    # Piecewise ground truth (kept private; the grader has an identical copy).
    # Mild regime for x0 below the break; an accelerating quadratic hinge
    # ("tipping" feedback) switches on smoothly at the break. Solvers must
    # rediscover this FORM from the data alone.
    C = 0.5                       # hidden regime-break location on x0
    hinge = x0 - C if x0 > C else 0.0
    base = 0.6 + 1.1 * x0 - 0.5 * x1 + 0.8 * x2 * x3 + 0.4 * x3
    return base + 1.5 * hinge * hinge


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        return 1
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    rng = random.Random(20260701 + 37 * t)
    n_rows = 260 - 12 * t          # 248 .. 140 rows (shrinks with difficulty)
    if n_rows < 80:
        n_rows = 80
    noise = 0.04 + 0.02 * t        # measurement noise (grows with difficulty)
    # near-regime data always visible; a shrinking (but never vanishing) slice
    # pokes past the break so the regime change stays *detectable* but is never
    # far-extrapolated here.
    x0_hi = 1.15 - 0.01 * t        # 1.14 .. 1.05 (upper edge of observed x0)

    out = []
    for _ in range(n_rows):
        x0 = rng.uniform(-1.0, x0_hi)
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        x3 = rng.uniform(-1.0, 1.0)
        y = hidden_law(x0, x1, x2, x3) + rng.gauss(0.0, noise)
        out.append("%.6f %.6f %.6f %.6f %.6f" % (x0, x1, x2, x3, y))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

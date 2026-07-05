#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample (stdin instance) to stdout.

Family: symbolic-regression (format E), theme: solar-farm inverters.
The rows below are noisy measurements sampled from a hidden 4-D "poly-exp"
inverter-response law over the operating region. The law, its coefficients and
the held-out EXTRAPOLATION split live ONLY inside the checker (verify.py) -- this
generator prints DATA ROWS ONLY, never the seed or the law.

Difficulty ladder (testId 1..10): more rows + slightly more measurement noise.
Input variables (all normalized to nominal operating point = 1.0):
  x0 = plane-of-array irradiance      x1 = module temperature rise
  x2 = DC bus voltage ratio           x3 = load / clipping fraction
Target y = per-string AC power surrogate (a.u.).
"""
import sys
import numpy as np


def _law(X):
    # HIDDEN ground-truth inverter-response law (also mirrored in verify.py).
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 5.0 * x0 * np.exp(-1.4 * x1) + 2.0 * x2**2 + 1.5 * x3**2 - 0.8 * x0 * x3 + 1.5


def main():
    t = int(sys.argv[1])
    rng = np.random.default_rng(90000 + t)          # seeded via testId only
    N = 300 + 130 * t                               # large-scale ladder: 430 .. 1600 rows
    x0 = rng.uniform(0.10, 1.00, N)
    x1 = rng.uniform(0.00, 1.00, N)
    x2 = rng.uniform(0.30, 1.00, N)
    x3 = rng.uniform(0.00, 1.00, N)
    X = np.stack([x0, x1, x2, x3], axis=1)
    sigma = 0.06 + 0.015 * t                        # measurement noise grows with the ladder
    y = _law(X) + rng.normal(0.0, sigma, N)

    out = [str(N)]
    for i in range(N):
        out.append("%.6f %.6f %.6f %.6f %.6f" % (x0[i], x1[i], x2[i], x3[i], y[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

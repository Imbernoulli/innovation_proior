#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE calibration sample (stdin instance) to stdout.

Family: hidden-affine-sparse-poly (format E), theme: decode a scrambled sensor's
calibration curve.

A sensor reports a raw reading `x`; the true physical quantity `y` is produced by
an UNKNOWN internal calibration: the electronics first apply a hidden affine
transform u = a*x + b (gain a, offset b), then a genuine physical response that is
a SPARSE polynomial S(u) = sum_k c_k * u^{e_k} (only a few active powers). You only
receive noisy (x, y) pairs recorded inside a NARROW calibration window. The hidden
affine map, the sparse coefficients/exponents and the held-out EXTRAPOLATION split
live ONLY inside the checker (verify.py); this generator prints DATA ROWS ONLY --
never the seed, the affine map, or the law.

Difficulty ladder (testId 1..10): more rows + slightly more measurement noise. The
window stays narrow, so the raw curve looks like a smooth low-order trend there even
though it is a high-degree polynomial after the hidden change of variables.
"""
import sys
import numpy as np

# ---- HIDDEN ground-truth calibration (mirrored byte-for-byte in verify.py) ----
A_TRUE, B_TRUE = 2.0, -1.0                 # hidden internal affine: u = a*x + b
TERMS = [(1.5, 4), (-2.0, 3), (1.0, 1)]    # sparse response S(u) = sum c_k u^{e_k}


def _law(x):
    u = A_TRUE * x + B_TRUE
    return sum(c * u ** e for c, e in TERMS)


def main():
    t = int(sys.argv[1])
    rng = np.random.default_rng(70000 + t)          # seeded via testId only
    N = 50 + 18 * t                                 # 68 .. 230 rows
    x = rng.uniform(0.0, 1.0, N)                    # NARROW calibration window
    sigma = 0.05 + 0.01 * t                         # measurement noise grows with ladder
    y = _law(x) + rng.normal(0.0, sigma, N)

    out = [str(N)]
    for i in range(N):
        out.append("%.6f %.6f" % (x[i], y[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

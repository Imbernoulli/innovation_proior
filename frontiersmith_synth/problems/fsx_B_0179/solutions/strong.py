# TIER: strong
# Discover the poly-exp structure and refit its coefficients by least squares:
#   y ~ b0 + b1 x1^2 + b2 x2 + b3 exp(0.6 x3) + b4 x1 x4 + b5 x2 x3 + b6 x4^2
# This basis matches the hidden law's shape, so it extrapolates far better than a
# linear model -- but coefficients estimated from noisy, finite train data still
# drift on the held-out region, and the complexity penalty applies, so it does not
# saturate.  Numbers are space-separated for a clean closed form.
import sys
import math
import numpy as np


def read_data():
    d = sys.stdin.read().split()
    n = int(d[0])
    vals = d[2:]
    rows = []
    Y = []
    for r in range(n):
        b = r * 5
        x1, x2, x3, x4 = (float(vals[b]), float(vals[b + 1]),
                          float(vals[b + 2]), float(vals[b + 3]))
        rows.append((x1, x2, x3, x4))
        Y.append(float(vals[b + 4]))
    return rows, np.array(Y)


def design(rows):
    A = []
    for x1, x2, x3, x4 in rows:
        A.append([1.0, x1 * x1, x2, math.exp(0.6 * x3),
                  x1 * x4, x2 * x3, x4 * x4])
    return np.array(A)


def main():
    rows, Y = read_data()
    A = design(rows)
    c, *_ = np.linalg.lstsq(A, Y, rcond=None)
    expr = (
        "%.6f + %.6f * x1 ** 2 + %.6f * x2 + %.6f * exp( 0.6 * x3 )"
        " + %.6f * x1 * x4 + %.6f * x2 * x3 + %.6f * x4 ** 2"
        % (c[0], c[1], c[2], c[3], c[4], c[5], c[6])
    )
    print(expr)


if __name__ == "__main__":
    main()

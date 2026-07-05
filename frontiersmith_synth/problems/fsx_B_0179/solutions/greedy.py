# TIER: greedy
# Least-squares LINEAR model  y ~ c0 + c1 x1 + c2 x2 + c3 x3 + c4 x4.
# Beats the constant baseline but ignores the quadratic / exp structure, so it
# extrapolates only moderately well.  Numbers are printed space-separated so the
# expression is a clean closed form.
import sys
import numpy as np


def read_data():
    d = sys.stdin.read().split()
    n = int(d[0])
    vals = d[2:]
    X = []
    Y = []
    for r in range(n):
        b = r * 5
        X.append([1.0, float(vals[b]), float(vals[b + 1]),
                  float(vals[b + 2]), float(vals[b + 3])])
        Y.append(float(vals[b + 4]))
    return np.array(X), np.array(Y)


def main():
    X, Y = read_data()
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    terms = ["%.6f" % c[0]]
    for i in range(1, 5):
        terms.append("%.6f * x%d" % (c[i], i))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()

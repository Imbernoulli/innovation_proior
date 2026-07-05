# TIER: greedy
# Ordinary least-squares AFFINE fit: y ~ w0*x0 + w1*x1 + w2*x2 + w3*x3 + b.
# Captures first-order structure but has no exp / quadratic terms, so it
# extrapolates poorly onto the held-out region. Beats trivial, far from strong.
import sys
import numpy as np


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    X = np.empty((n, 4)); y = np.empty(n)
    for i in range(n):
        X[i, 0] = float(next(it)); X[i, 1] = float(next(it))
        X[i, 2] = float(next(it)); X[i, 3] = float(next(it))
        y[i] = float(next(it))
    A = np.column_stack([X, np.ones(n)])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    expr = "%.6f * x0 + %.6f * x1 + %.6f * x2 + %.6f * x3 + %.6f" % (w[0], w[1], w[2], w[3], w[4])
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()

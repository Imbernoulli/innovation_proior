# TIER: strong
# Discovers the poly-exp FAMILY: fit y ~ c0*x0*exp(b*x1) + c1*x2^2 + c2*x3^2
#                                     + c3*x0*x3 + c4
# by grid-searching the single nonlinear rate b (deterministic), then solving
# the linear coefficients in closed form. Recovers the law -> low held-out RMSE,
# but the complexity penalty + irreducible noise keep it below saturation.
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
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    best = None
    for b in np.linspace(-2.0, 0.0, 41):
        F = np.column_stack([x0 * np.exp(b * x1), x2**2, x3**2, x0 * x3, np.ones(n)])
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        res = float(np.sqrt(np.mean((F @ c - y) ** 2)))
        if best is None or res < best[0]:
            best = (res, b, c)
    _, b, c = best
    expr = ("%.6f * x0 * exp( %.6f * x1 ) + %.6f * x2 ** 2 + %.6f * x3 ** 2 "
            "+ %.6f * x0 * x3 + %.6f"
            % (c[0], b, c[1], c[2], c[3], c[4]))
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()

# TIER: greedy
"""The obvious first move: treat this as ordinary time-series regression.
Fit a flexible degree-2 polynomial in the last two known years,
    x(t+1) ~= c0 + c1*x(t) + c2*x(t-1) + c3*x(t)^2 + c4*x(t)*x(t-1) + c5*x(t-1)^2
by ordinary least squares over the training window, and submit it as the
recurrence. This nails the training years (a flexible enough curve always
can, over a narrow window) but has no reason to respect any conserved
quantity, so nothing stops it drifting once rolled forward for a century."""
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = [float(v) for v in data[2:2 + n]]

    rows, ys = [], []
    for i in range(1, n - 1):
        x, xk1 = vals[i], vals[i - 1]
        rows.append([1.0, x, xk1, x * x, x * xk1, xk1 * xk1])
        ys.append(vals[i + 1])
    A = np.array(rows)
    y = np.array(ys)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c0, c1, c2, c3, c4, c5 = [float(c) for c in coef]

    expr = ("%.8f + %.8f*x + %.8f*xk1 + %.8f*x*x + %.8f*x*xk1 + %.8f*xk1*xk1"
             % (c0, c1, c2, c3, c4, c5))
    print("OUT %s" % expr)


if __name__ == "__main__":
    main()

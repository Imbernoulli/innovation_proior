# TIER: greedy
# The obvious recipe (Prony's method at a FIXED, small order): assume the
# sequence satisfies an order-2 constant-coefficient linear recurrence
# a(n) = c1*a(n-1) + c2*a(n-2), fit c1,c2 by least squares over the visible
# window, recover the two characteristic roots, and fit amplitudes.  This is
# the textbook approach and it looks great on paper -- but the visible window
# is dominated by ONE huge term, so a numerical order-2 fit just soaks up
# that single dominant behaviour (plus whatever curvature the residual data
# happens to show) and never asks "is order 2 actually enough?".  It never
# recovers the true (higher) order, so it extrapolates the WRONG generator
# far past the training window.
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    T, t = int(data[0]), int(data[1])
    rows = data[2:]
    vals = [0] * T
    for i in range(T):
        n = int(rows[2 * i])
        v = int(rows[2 * i + 1])
        vals[n] = float(v)

    X = np.array([[vals[i - 1], vals[i - 2]] for i in range(2, T)])
    b = np.array([vals[i] for i in range(2, T)])
    c1, c2 = np.linalg.lstsq(X, b, rcond=None)[0]

    disc = c1 * c1 + 4 * c2
    if disc >= 0:
        sq = disc ** 0.5
        r1 = (c1 + sq) / 2.0
        r2 = (c1 - sq) / 2.0
        if abs(r1 - r2) < 1e-9:
            r2 = r1 + 1e-6  # perturb to avoid a singular Vandermonde system
        M = np.array([[r1 ** (T - 2), r2 ** (T - 2)], [r1 ** (T - 1), r2 ** (T - 1)]])
        rhs = np.array([vals[T - 2], vals[T - 1]])
        A1, A2 = np.linalg.solve(M, rhs)
        print("( %r ) * ( %r ) ** n + ( %r ) * ( %r ) ** n" % (float(A1), float(r1), float(A2), float(r2)))
    else:
        # complex roots: the order-2 model degenerates; fall back to a plain
        # single dominant-root fit from the last two points.
        ratio = vals[T - 1] / vals[T - 2]
        A = vals[T - 1] / (ratio ** (T - 1))
        print("( %r ) * ( %r ) ** n" % (float(A), float(ratio)))


if __name__ == "__main__":
    main()

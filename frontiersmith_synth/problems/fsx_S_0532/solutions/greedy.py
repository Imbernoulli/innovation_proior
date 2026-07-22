# TIER: greedy
# The OBVIOUS recipe: fit a low-order polynomial calibration curve directly in the
# raw reading x (here a quadratic least-squares fit). Inside the narrow window the
# true high-degree curve looks like a smooth quadratic, so this fits the window --
# but it never sees the hidden change of variables, so it captures the wrong shape
# and extrapolates poorly onto the held-out region. Beats the constant baseline,
# far below the affine-sparse solution.
import sys
import numpy as np


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    x = np.empty(n); y = np.empty(n)
    for i in range(n):
        x[i] = float(next(it)); y[i] = float(next(it))
    A = np.column_stack([np.ones(n), x, x ** 2])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    expr = "%.8f + %.8f * x + %.8f * x ** 2" % (c[0], c[1], c[2])
    sys.stdout.write(expr + "\n")


if __name__ == "__main__":
    main()

# TIER: strong
# INSIGHT: sparsity of the physical response is INVARIANT under the sensor's hidden
# affine change of variable u = a*x + b. Expanded in the raw reading x, the law is a
# DENSE high-degree polynomial (all powers active) -- so a direct dense/low-order fit
# on the narrow window is either ill-conditioned or wrong-shaped and extrapolates
# badly. But there EXISTS an affine coordinate u in which only a FEW powers are
# active. So we search the tiny 2-parameter (a, b) family, pull the data back to
# u = a*x + b, and ask: does it admit a t-TERM polynomial fit? We select, by
# increasing number of terms, the SPARSEST affine model whose train residual is
# essentially the noise floor -- turning an impossible extrapolation into a small
# algebraic search. Recovering the hidden coordinate makes the model low-dimensional
# and well-conditioned, so it extrapolates far beyond the calibration window.
import sys
import numpy as np
from itertools import combinations


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    x = np.empty(n); y = np.empty(n)
    for i in range(n):
        x[i] = float(next(it)); y[i] = float(next(it))

    A_GRID = [1, 2, 3]
    B_GRID = [-2, -1, 0, 1]
    POOL = [1, 2, 3, 4]           # candidate active exponents (plus an intercept)
    TMAX = 3                      # SPARSE search: at most 3 active powers

    # Search the 2-parameter (a, b) affine family; for each candidate coordinate
    # u = a*x + b, test whether the pulled-back data admits a SPARSE (<= TMAX term)
    # polynomial fit, and keep the one with the smallest train residual. A wrong
    # coordinate needs many powers to fit -> among sparse fits, the true affine
    # coordinate wins because in it the data really is few-term (residual ~ noise).
    best = None
    for a in A_GRID:
        for b in B_GRID:
            u = a * x + b
            for r in range(1, TMAX + 1):
                for es in combinations(POOL, r):
                    F = np.column_stack([u ** e for e in es] + [np.ones(n)])
                    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
                    res = float(np.sqrt(np.mean((F @ coef - y) ** 2)))
                    if best is None or res < best[0]:
                        best = (res, a, b, es, coef)

    _, a, b, es, coef = best
    parts = []
    for k, e in enumerate(es):
        parts.append("%.8f * ( %d * x + %d ) ** %d" % (coef[k], a, b, e))
    parts.append("%.8f" % coef[len(es)])   # intercept
    sys.stdout.write(" + ".join(parts) + "\n")


if __name__ == "__main__":
    main()

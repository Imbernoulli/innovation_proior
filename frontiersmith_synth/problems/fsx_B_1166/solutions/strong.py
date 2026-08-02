# TIER: strong
"""Insight: the forward operator's null space is exactly what makes plain least squares
fail here -- committing to sparsity (per the stated <=S_MAX-active-cell prior) and paying
data-misfit for it recovers the true source instead of a smeared blob that merely fits
the visible wells. Implemented via orthogonal matching pursuit: greedily pick the single
grid cell whose forward column best explains the current visible-well residual, refit a
small non-negative least squares over the chosen support, repeat up to S_MAX times. This
is a reformulation (support search + low-dimensional refit), not "least squares with more
iterations" -- it exploits the sparsity prior the greedy tier ignores."""
import math
import sys

import numpy as np
from scipy.optimize import nnls


def green(x0, y0, xw, yw, t, vx, vy, D):
    dx = xw - x0 - vx * t
    dy = yw - y0 - vy * t
    denom = 4.0 * D * t
    return math.exp(-(dx * dx + dy * dy) / denom) / (math.pi * denom)


def main():
    toks = sys.stdin.read().split()
    pos = 0
    test_id = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    K = int(toks[pos]); pos += 1
    MT = int(toks[pos]); pos += 1
    S_MAX = int(toks[pos]); pos += 1
    D = float(toks[pos]); pos += 1
    vx = float(toks[pos]); pos += 1
    vy = float(toks[pos]); pos += 1
    times = [float(toks[pos + k]) for k in range(MT)]; pos += MT
    B_mass = float(toks[pos]); pos += 1

    wells = []
    readings = []
    for _ in range(K):
        row = int(toks[pos]); pos += 1
        col = int(toks[pos]); pos += 1
        r = [float(toks[pos + k]) for k in range(MT)]; pos += MT
        wells.append((row, col))
        readings.append(r)

    M = N * N
    cells = [(i, j) for i in range(N) for j in range(N)]

    A = np.zeros((K * MT, M))
    y = np.zeros(K * MT)
    r_idx = 0
    for (wr, wc), rvals in zip(wells, readings):
        xw, yw = wc + 0.5, wr + 0.5
        for tk, t in enumerate(times):
            for cidx, (ci, cj) in enumerate(cells):
                x0, y0 = cj + 0.5, ci + 0.5
                A[r_idx, cidx] = green(x0, y0, xw, yw, t, vx, vy, D)
            y[r_idx] = rvals[tk]
            r_idx += 1

    col_norms = np.linalg.norm(A, axis=0) + 1e-12
    resid = y.copy()
    support = []
    for _ in range(S_MAX):
        corr = (A.T @ resid) / col_norms
        for s in support:
            corr[s] = -1e18
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        support.append(j)
        Asub = A[:, support]
        coef, _ = nnls(Asub, y)
        resid = y - Asub @ coef

    x = np.zeros(M)
    if support:
        Asub = A[:, support]
        coef, _ = nnls(Asub, y)
        for s, c in zip(support, coef):
            x[s] = c

    s_tot = x.sum()
    cap = B_mass * 1.05
    if s_tot > cap and s_tot > 0:
        x = x * (cap / s_tot)

    sys.stdout.write(" ".join("%.8f" % v for v in x.tolist()) + "\n")


if __name__ == "__main__":
    main()

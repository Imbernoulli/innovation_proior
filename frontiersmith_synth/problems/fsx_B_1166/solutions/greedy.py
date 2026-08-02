# TIER: greedy
"""Obvious approach: fit the visible well readings with a plain (minimum-norm) least
squares solve of the linear forward model, clip negative rates to zero, and rescale
down only if the budget is exceeded. This is what "solve the linear system" naturally
gives you -- but the visible system is badly underdetermined (K*MT << M), so the
minimum-norm solution SMEARS mass across many correlated nearby cells instead of
committing to the true sparse source. It fits the visible wells well; it does not
localize the source or generalize to unseen wells."""
import math
import sys

import numpy as np


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

    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    cap = B_mass * 1.05
    if s > cap and s > 0:
        x = x * (cap / s)

    sys.stdout.write(" ".join("%.8f" % v for v in x.tolist()) + "\n")


if __name__ == "__main__":
    main()

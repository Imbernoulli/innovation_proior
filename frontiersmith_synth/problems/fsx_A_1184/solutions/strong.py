# TIER: strong
import sys
import numpy as np
from scipy.optimize import nnls
from scipy.special import erfc


def emg_basis(grid, mu, sigma, tau):
    if tau < 1e-9:
        z = (grid - mu) / sigma
        return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    z = grid - mu
    a = (sigma * sigma) / (2.0 * tau * tau) - z / tau
    a = np.clip(a, -700.0, 700.0)
    arg = sigma / (tau * np.sqrt(2.0)) - z / (sigma * np.sqrt(2.0))
    return np.exp(a) * erfc(arg) / (2.0 * tau)


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    testId = int(next(it)); T = int(next(it)); M = int(next(it))
    sigma = float(next(it)); thr = float(next(it)); lam = float(next(it))
    r = [int(next(it)) for _ in range(M)]
    trace = np.array([float(next(it)) for _ in range(T)])

    grid = np.arange(T, dtype=float)

    # KEY INSIGHT: every true peak in this run shares ONE physical column
    # tailing constant tau.  Instead of fitting each slot's shape independently
    # (which lets a heavy tail masquerade as a phantom neighboring compound),
    # grid-search a SINGLE shared tau and, for each candidate, jointly
    # non-negative-least-squares fit ALL M EMG basis columns against the whole
    # trace at once.  Keep the tau/fit with the smallest residual.
    best_x, best_resid = None, None
    for tau_c in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        Phi = np.zeros((T, M))
        for i, ri in enumerate(r):
            Phi[:, i] = emg_basis(grid, float(ri), sigma, tau_c)
        x, _ = nnls(Phi, trace)
        resid = float(np.sum((Phi @ x - trace) ** 2))
        if best_resid is None or resid < best_resid:
            best_resid = resid
            best_x = x

    print(" ".join("%.6f" % v for v in best_x))


main()

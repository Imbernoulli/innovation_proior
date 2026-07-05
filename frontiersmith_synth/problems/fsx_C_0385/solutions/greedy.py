# TIER: greedy
"""Single-pass linear imputation.  Start from per-channel mean imputation, then for
each channel fit ONE ridge regression on the OTHER (mean-filled) channels using the
rows where that channel is observed, and predict the holes -- a single sweep, no
iteration.  Better than plain mean imputation because it exploits cross-channel
correlation, but the predictors are frozen at their mean-filled values and the fit is
never refined, so it leaves a lot of recoverable signal on the table versus an
iterative method."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    n, d = int(inst["n"]), int(inst["d"])
    Xr = inst["X"]
    miss = inst["missing"]

    X = np.zeros((n, d), dtype=np.float64)
    obs = np.zeros((n, d), dtype=bool)
    for i in range(n):
        row = Xr[i]
        for j in range(d):
            v = row[j]
            if v is None:
                obs[i, j] = False
            else:
                X[i, j] = float(v)
                obs[i, j] = True

    col_mean = np.array([X[obs[:, j], j].mean() if obs[:, j].any() else 0.0
                         for j in range(d)], dtype=np.float64)

    # mean-filled working matrix (frozen predictors)
    F = X.copy()
    for j in range(d):
        F[~obs[:, j], j] = col_mean[j]

    lam = 1.0
    filled = F.copy()
    for j in range(d):
        others = [c for c in range(d) if c != j]
        rows = np.flatnonzero(obs[:, j])
        if rows.size < 3 or not others:
            filled[~obs[:, j], j] = col_mean[j]
            continue
        A = F[np.ix_(rows, others)]
        A = np.hstack([A, np.ones((A.shape[0], 1))])   # bias term
        b = X[rows, j]
        AtA = A.T @ A + lam * np.eye(A.shape[1])
        w = np.linalg.solve(AtA, A.T @ b)
        tgt = np.flatnonzero(~obs[:, j])
        if tgt.size:
            P = np.hstack([F[np.ix_(tgt, others)], np.ones((tgt.size, 1))])
            filled[tgt, j] = P @ w

    fill = [float(filled[i, j]) for (i, j) in miss]
    print(json.dumps(fill))


main()

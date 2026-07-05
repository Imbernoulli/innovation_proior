# TIER: strong
"""Iterative chained-equation imputation (MICE-style) with ridge regressors.

Initialize every hole with its channel mean, then repeatedly sweep the channels:
for each channel, ridge-regress it on ALL other channels using the rows where it is
observed, and OVERWRITE its holes with the fresh predictions.  Because the predictors
themselves are re-imputed on every sweep, the estimates co-adapt and converge toward
the low-rank structure of the sensor field -- recovering far more of the withheld
readings than a single pass or plain mean imputation on the reconstructable gardens,
while degrading gracefully to mean-level info on the near-full-rank noisy gardens.
That cross-regime robustness is what the geometric-mean objective rewards."""
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

    F = X.copy()
    for j in range(d):
        F[~obs[:, j], j] = col_mean[j]

    lam = 0.30
    n_sweeps = 12
    order = list(range(d))
    for _ in range(n_sweeps):
        for j in order:
            others = [c for c in range(d) if c != j]
            rows = np.flatnonzero(obs[:, j])
            tgt = np.flatnonzero(~obs[:, j])
            if tgt.size == 0:
                continue
            if rows.size < 3 or not others:
                F[tgt, j] = col_mean[j]
                continue
            A = np.hstack([F[np.ix_(rows, others)], np.ones((rows.size, 1))])
            b = X[rows, j]
            AtA = A.T @ A + lam * np.eye(A.shape[1])
            try:
                w = np.linalg.solve(AtA, A.T @ b)
            except np.linalg.LinAlgError:
                F[tgt, j] = col_mean[j]
                continue
            P = np.hstack([F[np.ix_(tgt, others)], np.ones((tgt.size, 1))])
            F[tgt, j] = P @ w

    fill = [float(F[i, j]) for (i, j) in miss]
    print(json.dumps(fill))


main()

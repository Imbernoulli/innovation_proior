# TIER: greedy
# Plain Lloyd k-means on the RAW features (single init, no preprocessing).
# Good on compact isotropic regimes; distorted by shear (aniso) and dominated by
# large-scale nuisance channels (scaled) -> collapses on those cities.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
X = np.asarray(inst["X"], dtype=float)
n = X.shape[0]
k = int(inst["k"])
rng = np.random.default_rng(int(inst["seed"]))


def lloyd(X, k, rng, n_iter=40):
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    lab = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for c in range(k):
            m = lab == c
            if m.any():
                C[c] = X[m].mean(axis=0)
    return lab


lab = lloyd(X, k, rng)
print(json.dumps(lab.tolist()))

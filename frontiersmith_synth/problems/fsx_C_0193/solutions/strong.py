# TIER: strong
# Robust, structure-adaptive clustering:
#   1) robust standardize (median / MAD) so wildly-scaled + nuisance channels are
#      put on comparable footing (fixes the "scaled" cities),
#   2) PCA whitening (decorrelate + unit-variance the principal axes) so sheared /
#      anisotropic regimes become round again (fixes the "aniso" cities),
#   3) k-means++ with several restarts, keep the lowest-inertia partition.
# Still imperfect on overlapping / many-regime / high-noise cities -> leaves
# headroom below a perfect recovery.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
X = np.asarray(inst["X"], dtype=float)
n, d = X.shape
k = int(inst["k"])
rng = np.random.default_rng(int(inst["seed"]))

# --- 1) robust per-feature standardization ---
med = np.median(X, axis=0)
mad = np.median(np.abs(X - med), axis=0)
scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
Z = (X - med) / scale

# --- 2) PCA whitening ---
Zc = Z - Z.mean(axis=0)
cov = (Zc.T @ Zc) / max(n - 1, 1)
evals, evecs = np.linalg.eigh(cov)
evals = np.clip(evals, 1e-8, None)
W = Zc @ (evecs / np.sqrt(evals))    # whitened coordinates


def kmpp_init(X, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    i0 = int(rng.integers(n))
    centers[0] = X[i0]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        tot = d2.sum()
        if tot <= 0:
            centers[c] = X[int(rng.integers(n))]
        else:
            probs = d2 / tot
            j = int(rng.choice(n, p=probs))
            centers[c] = X[j]
        nd = ((X - centers[c]) ** 2).sum(axis=1)
        d2 = np.minimum(d2, nd)
    return centers


def kmeans(X, k, rng, n_init=8, n_iter=60):
    best_lab, best_in = None, np.inf
    for _ in range(n_init):
        C = kmpp_init(X, k, rng)
        lab = np.zeros(X.shape[0], dtype=np.int64)
        d2 = None
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
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        inertia = d2[np.arange(X.shape[0]), lab].sum()
        if inertia < best_in:
            best_in, best_lab = inertia, lab
    return best_lab


lab = kmeans(W, k, rng)
print(json.dumps(lab.tolist()))

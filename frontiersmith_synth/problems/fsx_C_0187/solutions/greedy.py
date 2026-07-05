# TIER: greedy
"""Plain Lloyd k-means on z-standardized features (k-means++ init, a few seeded
restarts, best inertia).  Recovers compact, roughly isotropic segments well
(blobs / high-dim / mild anisotropy) but assumes convex equal-variance clusters,
so it slices interleaving crescents and concentric ability rings straight through
the middle -- the geometric mean punishes that collapse."""
import sys, json
import numpy as np


def _kmeans(Z, k, rng, iters=100):
    n = Z.shape[0]
    # k-means++ init
    c0 = rng.integers(0, n)
    centers = [Z[c0]]
    d2 = np.sum((Z - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers.append(Z[idx])
        nd = np.sum((Z - Z[idx]) ** 2, axis=1)
        d2 = np.minimum(d2, nd)
    C = np.array(centers, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        G = Z @ C.T
        sqZ = np.sum(Z ** 2, axis=1)[:, None]
        sqC = np.sum(C ** 2, axis=1)[None, :]
        D = sqZ + sqC - 2.0 * G
        new = np.argmin(D, axis=1)
        if np.array_equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = Z[m].mean(axis=0)
    inertia = 0.0
    for j in range(k):
        m = labels == j
        if m.any():
            inertia += float(np.sum((Z[m] - C[j]) ** 2))
    return labels, inertia


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X"], dtype=np.float64)
    k = int(inst["k"])
    seed = int(inst.get("seed", 0))
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd

    best_lab, best_in = None, float("inf")
    for r in range(6):
        rng = np.random.default_rng(seed + 100 * r + 1)
        lab, inertia = _kmeans(Z, k, rng)
        if inertia < best_in:
            best_in, best_lab = inertia, lab
    print(json.dumps(best_lab.tolist()))


main()

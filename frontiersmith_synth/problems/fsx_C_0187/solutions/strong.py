# TIER: strong
"""Normalized spectral clustering (Ng-Jordan-Weiss) on a self-tuning kNN affinity
graph, with k-means on the row-normalized leading eigenvectors of the symmetric
normalized Laplacian.

Because it clusters CONNECTIVITY in a robustly standardized space rather than raw
Euclidean distance to centroids, it recovers non-convex structure -- interleaving
beginner/expert crescents, concentric ability rings -- as well as compact blobs,
anisotropic diagonals, unequal-spread segments, and high-dimensional profiles.
That cross-geometry robustness is exactly what the geometric-mean objective
rewards; centroid-only methods collapse on the non-convex instances."""
import sys, json
import numpy as np


def _kmeans(Z, k, rng, iters=100):
    n = Z.shape[0]
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
    n = X.shape[0]
    k = int(inst["k"])
    seed = int(inst.get("seed", 0))

    # robust standardization (median / MAD)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    sd = np.where(mad > 1e-9, 1.4826 * mad, X.std(axis=0) + 1e-9)
    Zx = (X - med) / sd

    # pairwise squared distances
    G = Zx @ Zx.T
    sq = np.sum(Zx ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)

    # self-tuning affinity: local scale = distance to the m-th neighbour
    m = max(5, min(20, n // 12))
    Dsort = np.sort(D, axis=1)
    sigma = Dsort[:, m] + 1e-9               # local bandwidth per point
    A = np.exp(-D2 / (sigma[:, None] * sigma[None, :]))
    np.fill_diagonal(A, 0.0)

    # sparsify to a symmetric kNN graph (keep each point's m nearest)
    idx = np.argsort(D, axis=1)[:, 1:m + 1]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), m)
    mask[rows, idx.ravel()] = True
    mask = mask | mask.T
    A = A * mask

    # symmetric normalized Laplacian, take k smallest eigenvectors
    deg = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(deg + 1e-12)
    L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    L = 0.5 * (L + L.T)
    vals, vecs = np.linalg.eigh(L)
    U = vecs[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    U = U / norms

    best_lab, best_in = None, float("inf")
    for r in range(6):
        rng = np.random.default_rng(seed + 100 * r + 7)
        lab, inertia = _kmeans(U, k, rng)
        if inertia < best_in:
            best_in, best_lab = inertia, lab
    print(json.dumps(best_lab.tolist()))


main()

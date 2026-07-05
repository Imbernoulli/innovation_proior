# TIER: strong
# Transferable community detection: per-feature standardization + a symmetric
# k-nearest-neighbour affinity graph, normalized-Laplacian spectral embedding, and
# k-means on the row-normalized eigenvectors. The graph/connectivity view captures
# ring, chain and concentric transmission that Euclidean k-means cannot, while still
# recovering plain blobs. Fully deterministic (fixed neighbourhood + fixed seeding),
# no wall-clock, no external ML libraries.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
X = np.asarray(inst["X"], dtype=float)
k = int(inst["k"])
n = len(X)

# 1) standardize so anisotropic / differently-scaled exposures are comparable
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd < 1e-9] = 1.0
Z = (X - mu) / sd

# 2) pairwise distances + symmetric kNN affinity with a local Gaussian scale
D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
nn = max(5, min(n - 1, int(round(np.log(n) * 3))))
order = np.argsort(D2, axis=1)
knn = order[:, 1:nn + 1]
# local scale = distance to the nn-th neighbour (self-tuning affinity)
sigma = np.sqrt(np.maximum(D2[np.arange(n), knn[:, -1]], 1e-12))
W = np.zeros((n, n))
for i in range(n):
    j = knn[i]
    w = np.exp(-D2[i, j] / (sigma[i] * sigma[j] + 1e-12))
    W[i, j] = w
W = np.maximum(W, W.T)            # symmetrize

# 3) normalized Laplacian L_sym = I - D^-1/2 W D^-1/2
deg = W.sum(axis=1)
deg[deg < 1e-12] = 1e-12
dinv = 1.0 / np.sqrt(deg)
L = np.eye(n) - (dinv[:, None] * W * dinv[None, :])
L = 0.5 * (L + L.T)

# 4) smallest-k eigenvectors -> spectral embedding, row-normalized
vals, vecs = np.linalg.eigh(L)
U = vecs[:, :k]
rn = np.sqrt((U ** 2).sum(axis=1))
rn[rn < 1e-12] = 1.0
U = U / rn[:, None]

# 5) k-means on the embedding (deterministic k-means++)
rng = np.random.RandomState(0)


def kmeanspp(P, k, rng):
    m = len(P)
    idx = [int(rng.randint(m))]
    d2 = ((P - P[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        pr = d2 / max(d2.sum(), 1e-12)
        nx = int(rng.choice(m, p=pr))
        idx.append(nx)
        d2 = np.minimum(d2, ((P - P[nx]) ** 2).sum(axis=1))
    return P[idx].copy()


def kmeans(P, k, rng, iters=100):
    C = kmeanspp(P, k, rng)
    lab = np.zeros(len(P), dtype=int)
    for _ in range(iters):
        DD = ((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = DD.argmin(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            msk = lab == j
            if msk.any():
                C[j] = P[msk].mean(axis=0)
    inertia = ((P - C[lab]) ** 2).sum()
    return lab, inertia

# a few deterministic restarts, keep lowest inertia
best_lab, best_in = None, np.inf
for seed in range(6):
    lab, inert = kmeans(U, k, np.random.RandomState(seed))
    if inert < best_in:
        best_in, best_lab = inert, lab

print(json.dumps({"labels": [int(v) for v in best_lab]}))

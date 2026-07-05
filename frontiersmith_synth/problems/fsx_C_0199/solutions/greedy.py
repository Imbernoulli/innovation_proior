# TIER: greedy
# Textbook Lloyd k-means on the raw exposure coordinates with a deterministic
# k-means++ seeding. Recovers round, well-separated household blobs, but its
# spherical/Euclidean bias mislabels ring, chain and concentric transmission
# geometries -> collapses on the non-convex datasets.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
X = np.asarray(inst["X"], dtype=float)
k = int(inst["k"])
rng = np.random.RandomState(0)
n = len(X)


def kmeanspp(X, k, rng):
    idx = [int(rng.randint(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        nxt = int(rng.choice(n, p=probs))
        idx.append(nxt)
        d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(axis=1))
    return X[idx].copy()


C = kmeanspp(X, k, rng)
labels = np.zeros(n, dtype=int)
for _ in range(100):
    D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    new = D.argmin(axis=1)
    if np.array_equal(new, labels):
        break
    labels = new
    for j in range(k):
        m = labels == j
        if m.any():
            C[j] = X[m].mean(axis=0)

print(json.dumps({"labels": [int(v) for v in labels]}))

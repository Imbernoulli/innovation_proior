# TIER: strong
"""Local-density anomaly score: mean distance to the k nearest neighbours in a
robustly standardized feature space (median / MAD).  Because it measures LOCAL
sparsity rather than distance to a single global centroid, it adapts to every
fleet structure -- far global outliers, correlation-manifold breakers (off the
manifold -> few near neighbours), multi-modal regimes (dense within a mode),
scaled features (standardized away), and local outliers just off a dense
cluster -- so it stays strong across the whole family, which the geometric mean
rewards."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X"], dtype=np.float64)
    n = X.shape[0]

    # robust per-feature standardization (median / MAD) -> scale invariance
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    sd = np.where(mad > 1e-9, 1.4826 * mad, X.std(axis=0) + 1e-9)
    Z = (X - med) / sd

    k = max(3, min(15, n // 12))
    # pairwise squared distances (small N per fleet)
    G = Z @ Z.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, np.inf)

    # mean distance to the k nearest neighbours
    part = np.partition(D2, k - 1, axis=1)[:, :k]
    scores = np.sqrt(part).mean(axis=1)
    print(json.dumps(scores.tolist()))


main()

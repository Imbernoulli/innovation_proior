# TIER: greedy
"""Greedy reducer: plain PCA -- centre by the training mean and keep the top-d
principal directions (highest raw variance).  Excellent when the cell-type
signal is the dominant variance (clean atlases), but on noisy / wildly-scaled
atlases the top variance directions are loud technical-noise or high-expression
housekeeping genes, so the embedding is distracted and kNN degrades."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X_train"], dtype=np.float64)
    d = int(inst["d_target"])
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d]
    print(json.dumps({"W": W.tolist(), "b": mu.tolist()}))


main()

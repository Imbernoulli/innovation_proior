# TIER: trivial
"""Trivial reducer: a random Gaussian (Johnson-Lindenstrauss style) projection
centred on the training mean.  This mirrors the evaluator's internal random
baseline, so on every atlas it normalizes to ~0.1."""
import sys, json, math
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X_train"], dtype=np.float64)
    D = int(inst["n_genes"])
    d = int(inst["d_target"])
    rng = np.random.default_rng(int(inst["seed"]) + 7)
    W = rng.normal(0.0, 1.0 / math.sqrt(D), size=(d, D))
    b = X.mean(axis=0)
    print(json.dumps({"W": W.tolist(), "b": b.tolist()}))


main()

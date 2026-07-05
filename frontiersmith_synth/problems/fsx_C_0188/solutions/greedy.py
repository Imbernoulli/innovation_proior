# TIER: greedy
"""Greedy discovery: MARGINAL-correlation skeleton + variance orientation.

Skeleton: connect every intersection pair whose absolute Pearson correlation
exceeds a fixed threshold.  Orientation: in the equal-variance regime an upstream
(source) intersection has SMALLER marginal variance than its downstream feeders,
so orient each retained edge low-variance -> high-variance.

This recovers a lot of true structure but uses MARGINAL correlation, which also
fires on indirect (transitively induced) associations -> it adds spurious edges
between non-adjacent intersections, capping its accuracy well below a routine that
conditions on the already-explained congestion."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["data"], dtype=np.float64)
    p = int(inst["n_nodes"])

    var = X.var(axis=0) + 1e-12
    # correlation matrix
    Xc = X - X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    Z = Xc / std
    C = (Z.T @ Z) / X.shape[0]

    thr = 0.30
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if abs(C[i, j]) > thr:
                if var[i] <= var[j]:
                    edges.append([i, j])
                else:
                    edges.append([j, i])
    print(json.dumps({"edges": edges}))


main()

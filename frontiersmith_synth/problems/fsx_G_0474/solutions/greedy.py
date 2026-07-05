# TIER: greedy
"""Marginal-correlation thresholding, orientation by raw column order.

Add an (undirected) edge between every pair of channels whose absolute Pearson
correlation exceeds a fixed threshold, then orient it by raw index order
(i -> j for i < j).  This picks up real adjacencies but also every INDIRECT
correlation along a causal path (false positives), and because the mesh channels
are randomly relabelled the index-order orientation is essentially a coin flip --
so it recovers some skeleton yet pays heavily in extra and reversed edges.  It
ignores the published forcing-node anchor entirely."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["data"], dtype=np.float64)
    d = int(inst["n_nodes"])
    C = np.corrcoef(X, rowvar=False)
    thr = 0.30
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if abs(C[i, j]) > thr:
                edges.append([i, j])           # orient by index order (uninformed)
    print(json.dumps({"edges": edges}))


main()

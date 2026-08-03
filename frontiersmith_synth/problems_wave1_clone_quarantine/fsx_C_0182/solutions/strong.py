# TIER: strong
"""Precision-matrix (partial-correlation) skeleton + equal-variance ordering.

Skeleton: the inverse covariance (precision) matrix is (up to scaling) the matrix
of partial correlations; an off-diagonal entry near zero means the two channels
are conditionally independent given all the others.  Thresholding |partial corr|
recovers the conditional-independence graph, which is the true skeleton plus the
"moralization" edges between co-parents -- far cleaner than marginal correlation
because it removes indirect-path false edges.

Orientation: under equal-variance disturbances a source channel has smaller
marginal variance than its descendants, so sorting channels by increasing
variance estimates a topological order; each skeleton edge is oriented from the
lower-variance (earlier) channel to the higher-variance (later) one.

Leftover moralization edges and the occasional mis-ordered pair keep it short of
perfect, especially on the larger, data-poorer arrays."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["data"], dtype=np.float64)
    d = int(inst["n_nodes"])

    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    cov = cov + 1e-6 * np.eye(d)
    prec = np.linalg.pinv(cov)
    dp = np.sqrt(np.abs(np.diag(prec))) + 1e-12
    pcorr = -prec / np.outer(dp, dp)      # partial correlation matrix

    var = X.var(axis=0)
    order = np.argsort(var, kind="mergesort")     # ascending variance ~ topo order
    rank = np.empty(d, dtype=np.int64)
    for pos, node in enumerate(order):
        rank[node] = pos

    thr = 0.10
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if abs(pcorr[i, j]) > thr:
                if rank[i] < rank[j]:
                    edges.append([i, j])
                else:
                    edges.append([j, i])
    print(json.dumps({"edges": edges}))


main()

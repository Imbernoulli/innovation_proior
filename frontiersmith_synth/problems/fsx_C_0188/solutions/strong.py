# TIER: strong
"""Strong discovery: equal-variance topological ordering + regression pruning.

Step 1 (causal order).  In an equal-variance linear-Gaussian SCM the true source
has the smallest residual variance.  Greedily build a topological order: at each
step pick the not-yet-ordered intersection whose residual variance -- after
regressing it on the ALREADY-ordered intersections -- is smallest, append it, and
repeat.  This yields an upstream->downstream order without ever seeing the labels.

Step 2 (parents by conditioning).  For each intersection j (in order) regress its
congestion on ALL its predecessors and keep as parents only those with a
standardized coefficient above a threshold.  Conditioning on the predecessors
removes the transitively-induced (indirect) associations that fool a marginal-
correlation skeleton, so the recovered graph is both better oriented AND far
cleaner -- giving a genuinely low SHD across grids.  Finite snapshots, dense flow
and large grids still leave residual errors, so it does not saturate."""
import sys, json
import numpy as np


def _resid_var(y, P):
    """Variance of y after least-squares regression on columns of P (with bias)."""
    if P.shape[1] == 0:
        return float(np.var(y))
    A = np.hstack([P, np.ones((P.shape[0], 1))])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    return float(np.var(res))


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["data"], dtype=np.float64)
    T, p = X.shape
    p = int(inst["n_nodes"])

    remaining = list(range(p))
    order = []
    while remaining:
        best, best_v = None, None
        Pcols = X[:, order] if order else np.zeros((T, 0))
        for c in remaining:
            v = _resid_var(X[:, c], Pcols)
            if best_v is None or v < best_v - 1e-12:
                best_v, best = v, c
        order.append(best)
        remaining.remove(best)

    thr = 0.12
    edges = []
    sd = X.std(axis=0) + 1e-12
    for k, j in enumerate(order):
        preds = order[:k]
        if not preds:
            continue
        P = X[:, preds]
        A = np.hstack([P, np.ones((T, 1))])
        beta, _, _, _ = np.linalg.lstsq(A, X[:, j], rcond=None)
        coef = beta[:-1]
        for m, i in enumerate(preds):
            std_coef = coef[m] * sd[i] / sd[j]
            if abs(std_coef) > thr:
                edges.append([i, j])
    print(json.dumps({"edges": edges}))


main()

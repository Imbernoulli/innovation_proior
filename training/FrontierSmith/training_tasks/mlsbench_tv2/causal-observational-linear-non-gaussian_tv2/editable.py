# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (order-first directed recovery): commit to ONE
# global topological order, then select edges by regressing each
# variable on its predecessors. The placeholder builds the order from
# a crude aggregate of pairwise cube-cumulant contrasts -- a noisy
# permutation statistic with no contradiction resolution and no
# iterative re-estimation. Better ordering (DirectLiNGAM-style root
# extraction, residual-independence scoring) and better edge selection
# along the order are the headroom. Order mistakes reverse whole
# fan-ins at once, so the order is where effort belongs.
# =====================================================================
def _global_order(Z: np.ndarray) -> np.ndarray:
    """Sort variables by aggregate upstream evidence (most causal first)."""
    m = Z.shape[0]
    C = (Z.T @ Z) / m
    M3 = ((Z ** 3).T @ Z) / m          # M3[i, j] = E[zi^3 zj]
    R = C * (M3.T - M3)                # R[i, j] > 0: evidence i upstream of j
    return np.argsort(-R.sum(axis=1))


def run_causal_discovery(X: np.ndarray, coef_thresh: float = 0.15) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    Placeholder pipeline: (1) one global order from _global_order;
    (2) least-squares regression of every variable on ALL predecessors
    in that order; (3) keep coefficients with |b| > coef_thresh. The
    order is never revisited -- by design, so its quality is exposed.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    Z /= np.where(sd > 0, sd, 1.0)

    order = _global_order(Z)
    B = np.zeros((d, d))
    for pos in range(1, d):
        node = int(order[pos])
        preds = order[:pos].astype(int)
        beta, *_ = np.linalg.lstsq(Z[:, preds], Z[:, node], rcond=None)
        for c, b in zip(preds, beta):
            if abs(float(b)) > coef_thresh:
                B[node, int(c)] = float(b)
    return B
# =====================================================================

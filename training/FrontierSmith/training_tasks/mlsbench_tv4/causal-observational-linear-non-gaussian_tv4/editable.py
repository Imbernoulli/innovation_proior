# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (dense-graph recall under FP control): reach the
# direct edges that marginal-correlation screens miss by admitting
# pairs on PARTIAL-correlation support, and orient every admitted pair
# -- abstention is not allowed in this variant, so admission quality is
# everything. The placeholder's orientation is a naive third-order skew
# contrast that goes quiet on symmetric noise (Laplace, uniform);
# replacing it, and making the admission rule properly regularized
# (shrinkage, sparse inverse covariance), is the headroom.
# =====================================================================
def _skew_orient(zi: np.ndarray, zj: np.ndarray) -> bool:
    """True if third-order skew evidence points i -> j (naive placeholder)."""
    return float(np.mean(zi ** 2 * zj) - np.mean(zi * zj ** 2)) >= 0.0


def run_causal_discovery(X: np.ndarray, pc_thresh: float = 0.08) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    Admission runs on the pseudo-inverse partial-correlation matrix so
    conditional (direct) association decides membership; every admitted
    pair is then oriented and reported with its partial correlation as
    the weight. ``pc_thresh`` is the single admission knob.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    Z /= np.where(sd > 0, sd, 1.0)

    C = np.clip(np.corrcoef(Z, rowvar=False), -0.999999, 0.999999)
    P = np.linalg.pinv(C)
    s = np.sqrt(np.abs(np.diag(P)))
    s = np.where(s > 0, s, 1.0)
    pcorr = -P / np.outer(s, s)
    np.fill_diagonal(pcorr, 0.0)

    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            w = float(pcorr[i, j])
            if abs(w) <= pc_thresh:
                continue
            if _skew_orient(Z[:, i], Z[:, j]):
                B[j, i] = w   # i -> j
            else:
                B[i, j] = w   # j -> i
    return B
# =====================================================================

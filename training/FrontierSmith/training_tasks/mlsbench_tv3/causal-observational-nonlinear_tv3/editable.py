# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (arrow-first ANM orientation): the adjacency step is
# deliberately unremarkable; the contribution is the two-way residual
# contrast that orients each kept link, its confidence margin, and the
# below-margin policy (here: drop the edge). Replace the moving-average
# smoother and the moment-proxy dependence measure with something
# sharper -- that is where this variant's headroom lives.
# =====================================================================
def _smooth_residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual of y after a centered moving-average fit along sorted x
    (window ~ sqrt(n)); a cheap stand-in for a proper 1-D smoother."""
    n = x.shape[0]
    w = max(5, int(np.sqrt(n)))
    o = np.argsort(x)
    ys = y[o]
    c = np.concatenate(([0.0], np.cumsum(ys)))
    idx = np.arange(n)
    lo = np.clip(idx - w // 2, 0, n)
    hi = np.clip(idx + w // 2 + 1, 0, n)
    mu = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    r = np.empty(n)
    r[o] = ys - mu
    return r


def _residual_dependence(x: np.ndarray, r: np.ndarray) -> float:
    """How much the residual r still knows about the regressor x.
    Spearman association of x with r and with |r|; ~0 under independence."""
    def _rank(a):
        q = np.argsort(np.argsort(a)).astype(np.float64)
        return (q - q.mean()) / (q.std() + 1e-12)
    rx = _rank(x)
    return abs(float(np.mean(rx * _rank(r)))) \
        + abs(float(np.mean(rx * _rank(np.abs(r)))))


def run_causal_discovery(X: np.ndarray,
                         assoc_floor: float = 0.15,
                         margin: float = 0.02) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    An arrow i -> j is emitted only when the residual of j-given-i is
    cleaner than the residual of i-given-j by more than ``margin``;
    otherwise the pair is abstained on (edge dropped). ``assoc_floor``
    is the plain rank-correlation cutoff for candidate links.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # unremarkable skeleton: thresholded Spearman association
    Rk = np.argsort(np.argsort(Z, axis=0), axis=0).astype(np.float64)
    S = np.corrcoef(Rk, rowvar=False)
    np.fill_diagonal(S, 0.0)

    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if abs(S[i, j]) < assoc_floor:
                continue
            r_j = _smooth_residual(Z[:, i], Z[:, j])   # candidate i -> j
            r_i = _smooth_residual(Z[:, j], Z[:, i])   # candidate j -> i
            dep_fwd = _residual_dependence(Z[:, i], r_j)
            dep_rev = _residual_dependence(Z[:, j], r_i)
            if dep_rev - dep_fwd > margin:
                B[j, i] = 1.0        # confident: i -> j
            elif dep_fwd - dep_rev > margin:
                B[i, j] = 1.0        # confident: j -> i
            # else: abstain -- weak asymmetry, claim nothing
    return B
# =====================================================================

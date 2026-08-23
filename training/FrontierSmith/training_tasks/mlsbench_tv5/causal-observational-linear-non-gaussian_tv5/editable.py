# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (tail-tamed higher-order statistics): every moment
# beyond second order is computed on an influence-bounded copy of the
# data -- median/MAD standardization followed by hard clipping -- so a
# few extreme rows cannot own the orientation decision. The clip level
# is a frozen constant here; calibrating it from measured tail weight
# (so uniform noise is left untouched while Laplace rows are tamed),
# and swapping the cube contrast for a bounded-influence contrast
# function, is the headroom.
# =====================================================================
def _tail_tamed(X: np.ndarray, clip: float) -> np.ndarray:
    """Median/MAD standardization, then symmetric clipping."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    mad = np.where(mad > 1e-12, mad, 1.0)
    return np.clip((X - med) / mad, -clip, clip)


def run_causal_discovery(X: np.ndarray, corr_thresh: float = 0.25,
                         clip: float = 3.0) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    Skeleton and orientation both run on the tamed matrix: correlation
    threshold for adjacency, then a vectorized cube-cumulant contrast
    (whose inputs are bounded by the clip, hence outlier-resistant)
    for direction.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    W = _tail_tamed(X, clip)

    C = np.corrcoef(W, rowvar=False)
    np.fill_diagonal(C, 0.0)
    M3 = ((W ** 3).T @ W) / n          # M3[i, j] = E[wi^3 wj] on tamed data
    D = M3.T - M3                      # D[i, j] = E[wi wj^3] - E[wi^3 wj]

    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            r = float(C[i, j])
            if abs(r) <= corr_thresh:
                continue
            if r * float(D[i, j]) > 0.0:
                B[j, i] = r   # evidence for i -> j
            else:
                B[i, j] = r   # evidence for j -> i
    return B
# =====================================================================

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (precision-governed directed recovery): emit an edge
# ONLY with directional evidence from non-Gaussianity; the reporting
# budget scales with d (not with the number of correlated pairs) so the
# 100-node hub graph cannot flood the output. The direction statistic
# must not assume a noise family (Laplace / exponential / uniform all
# appear, i.e. super-Gaussian, skewed, and sub-Gaussian).
# =====================================================================
def _direction_statistic(zi: np.ndarray, zj: np.ndarray) -> float:
    """Pairwise cause-effect asymmetry from higher-order moments.

    Placeholder: the cube-cumulant contrast E[zi*zj^3] - E[zi^3*zj]
    (Hyvarinen-Smith style). Its sign convention flips between super-
    and sub-Gaussian noise -- making the measure noise-family-agnostic
    (likelihood-ratio or entropy based) is exactly this variant's
    headroom.
    """
    return float(np.mean(zi * zj ** 3) - np.mean(zi ** 3 * zj))


def run_causal_discovery(X: np.ndarray,
                         edges_per_node: float = 1.0,
                         orient_margin: float = 0.0) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    ``edges_per_node`` caps emitted edges at edges_per_node * d (the
    reporting budget). ``orient_margin`` is the admission ticket: pairs
    whose |direction statistic| is <= margin count as unorientable and
    are NOT reported, so unsupported arrows never reach the metric.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    Z /= np.where(sd > 0, sd, 1.0)
    C = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(C, 0.0)

    iu = np.triu_indices(d, k=1)
    order = np.argsort(np.abs(C[iu]))[::-1][:int(round(edges_per_node * d))]

    B = np.zeros((d, d))
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        r = float(C[i, j])
        if r == 0.0:
            break
        stat = r * _direction_statistic(Z[:, i], Z[:, j])
        if abs(stat) <= orient_margin:
            continue  # unorientable => unreported
        if stat > 0:   # evidence for i -> j
            B[j, i] = r
        else:          # evidence for j -> i
            B[i, j] = r
    return B
# =====================================================================

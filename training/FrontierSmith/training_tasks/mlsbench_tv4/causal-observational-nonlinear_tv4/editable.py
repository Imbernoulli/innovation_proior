# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (worst-setting robustness): one fixed pipeline whose
# every statistic is distribution-free or median-based, so that no
# noise family or sample size is silently favored. Degree caps and a
# conservative orientation ratio protect the weakest setting; every
# "constant" that could differ between regimes is a continuous function
# of measurable quantities (n only enters through 1/sqrt(n) and bin
# counts). Sharpen the robust statistics; do not specialize them.
# =====================================================================
def _robust_scale(X: np.ndarray) -> np.ndarray:
    """Median/MAD standardization followed by winsorization at +-2.5:
    heavy tails (Laplace, exponential) cannot dominate any later moment."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-12
    Z = (X - med) / mad
    return np.clip(Z, -2.5, 2.5)


def _binned_iqr(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    """IQR of b around its per-bin median along quantile bins of a --
    a robust residual-spread proxy for the regression b ~ f(a)."""
    edges = np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, n_bins - 1)
    resid = b.copy()
    for c in range(n_bins):
        m = idx == c
        if m.any():
            resid[m] = b[m] - np.median(b[m])
    q75, q25 = np.quantile(resid, [0.75, 0.25])
    return float(q75 - q25) + 1e-12


def run_causal_discovery(X: np.ndarray,
                         degree_cap: int = 3,
                         orient_ratio: float = 0.97) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    Skeleton: Spearman association above a 1/sqrt(n) null band, admitted
    in decreasing strength subject to ``degree_cap`` per node. Arrows:
    the direction whose robust residual spread is smaller by factor
    ``orient_ratio``; near-symmetric pairs are left unclaimed.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = _robust_scale(X)
    n_bins = int(np.clip(round(n ** 0.4), 4, 14))   # continuous in n

    Rk = np.argsort(np.argsort(Z, axis=0), axis=0).astype(np.float64)
    S = np.corrcoef(Rk, rowvar=False)
    np.fill_diagonal(S, 0.0)
    thr = 2.6 / np.sqrt(n)                          # rank-corr null band

    iu = np.triu_indices(d, k=1)
    strength = np.abs(S[iu])
    order = np.argsort(strength)[::-1]
    deg = np.zeros(d, dtype=int)
    B = np.zeros((d, d))
    for k in order:
        if strength[k] <= thr:
            break
        i, j = int(iu[0][k]), int(iu[1][k])
        if deg[i] >= degree_cap or deg[j] >= degree_cap:
            continue
        spread_fwd = _binned_iqr(Z[:, i], Z[:, j], n_bins)  # i -> j
        spread_rev = _binned_iqr(Z[:, j], Z[:, i], n_bins)  # j -> i
        if spread_fwd < orient_ratio * spread_rev:
            B[j, i] = 1.0
        elif spread_rev < orient_ratio * spread_fwd:
            B[i, j] = 1.0
        else:
            continue                                # too symmetric: no claim
        deg[i] += 1
        deg[j] += 1
    return B
# =====================================================================

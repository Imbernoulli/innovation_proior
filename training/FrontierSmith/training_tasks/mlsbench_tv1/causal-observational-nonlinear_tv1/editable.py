# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (budgeted, stability-gated ANM discovery): a single
# configuration must serve n=150 and n=2000. Every regression is paid
# from an explicit fit budget, and an oriented edge is reported only if
# the SAME claim recurs across independent subsamples.
# =====================================================================
def _anm_asymmetry(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Directional score via binned-mean regression in both directions.

    Under an additive-noise model x -> y, predicting y from x leaves a
    smaller residual variance than predicting x from y. Coarse on
    purpose -- replace with proper nonparametric regression plus
    residual-independence testing; this is the variant's headroom.
    Returns > 0 when the evidence favors x -> y.
    """
    def _resid_var(a, b):
        edges = np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, n_bins - 1)
        mu = np.zeros(n_bins)
        for c in range(n_bins):
            m = idx == c
            if m.any():
                mu[c] = b[m].mean()
        return float(np.var(b - mu[idx]))

    return _resid_var(y, x) - _resid_var(x, y)


def run_causal_discovery(X: np.ndarray,
                         n_subsamples: int = 4,
                         stability_threshold: float = 0.75,
                         fit_budget: int = 3000) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    A directed edge enters B only if the same oriented claim appears in
    at least ``stability_threshold`` of ``n_subsamples`` 80% subsamples
    (the stability gate). ``fit_budget`` caps the total number of binned
    regressions across ALL subsamples (the compute budget); bin counts
    adapt continuously to the subsample size, never by branching on n.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    Z /= np.where(sd > 0, sd, 1.0)
    rng = np.random.default_rng(0)

    budget = [int(fit_budget)]  # mutable counter shared by all subsamples

    def _vote(Zs: np.ndarray) -> np.ndarray:
        ns = Zs.shape[0]
        n_bins = int(np.clip(np.sqrt(ns) / 2.0, 3, 12))  # continuous in ns
        # Rank transform -> Spearman association, robust to monotone
        # nonlinearity; the weak spot is non-monotone links.
        R = np.argsort(np.argsort(Zs, axis=0), axis=0).astype(float)
        C = np.corrcoef(R, rowvar=False)
        np.fill_diagonal(C, 0.0)
        iu = np.triu_indices(d, k=1)
        order = np.argsort(np.abs(C[iu]))[::-1][:d]  # ~1 candidate per node
        A = np.zeros((d, d))
        for k in order:
            if budget[0] < 2:
                break  # budget exhausted: stop claiming edges
            i, j = int(iu[0][k]), int(iu[1][k])
            if abs(C[i, j]) < 0.05:
                break
            budget[0] -= 2  # two binned regressions per pair
            s = _anm_asymmetry(Zs[:, i], Zs[:, j], n_bins)
            if s > 0:
                A[j, i] = 1.0   # i -> j
            elif s < 0:
                A[i, j] = 1.0   # j -> i
        return A

    votes = np.zeros((d, d))
    for _ in range(int(n_subsamples)):
        idx = rng.choice(n, size=max(2, int(0.8 * n)), replace=False)
        votes += _vote(Z[idx])
    B = (votes / max(1, int(n_subsamples)) >= stability_threshold).astype(float)
    return B
# =====================================================================

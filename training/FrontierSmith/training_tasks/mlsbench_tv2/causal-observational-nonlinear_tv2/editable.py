# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (calibrated skeleton precision): adjacency claims
# come from NONPARAMETRIC dependence statistics whose acceptance
# thresholds are derived from a permutation null on the data at hand,
# with a multiplicity correction over all candidate pairs, and are
# screened by conditional checks before orientation. The placeholder
# below is deliberately coarse; its calibration and its conditional
# screen are the intended headroom.
# =====================================================================
def _perm_threshold(R: np.ndarray, n_perm: int, rng) -> float:
    """(1 - 1/d^2)-style null quantile of |rank correlation| under
    independence, estimated by column permutations of the observed data."""
    n, d = R.shape
    null = np.empty(n_perm)
    for t in range(n_perm):
        i, j = rng.integers(0, d, size=2)
        p = rng.permutation(n)
        a, b = R[:, i], R[p, j]
        null[t] = abs(np.corrcoef(a, b)[0, 1])
    # crude Bonferroni-flavored inflation for d*(d-1)/2 simultaneous tests
    return float(np.quantile(null, 0.98) * 1.5)


def _poly_resid_dependence(z: np.ndarray, y: np.ndarray) -> float:
    """Dependence of the residual on the regressor after a cubic fit.
    Small value = residual looks independent = plausible causal direction."""
    coef = np.polyfit(z, y, 3)
    r = y - np.polyval(coef, z)
    r = (r - r.mean()) / (r.std() + 1e-12)
    z2 = z * z - (z * z).mean()
    return abs(float(np.mean(z * r))) + abs(float(np.mean(z2 * r))) \
        + abs(float(np.mean(z * (r * r - 1.0))))


def run_causal_discovery(X: np.ndarray, n_perm: int = 60) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    Pipeline: (1) rank-transform; (2) permutation-calibrated marginal
    dependence screen; (3) order-1 partial-rank screen against common
    neighbors (precision guard vs. shared causes); (4) routine
    cubic-residual orientation of the surviving adjacencies.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(0)
    # rank transform: distribution-free marginals
    R = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-12)

    thr = _perm_threshold(R, n_perm, rng)
    C = np.corrcoef(R, rowvar=False)
    np.fill_diagonal(C, 0.0)
    adj = np.abs(C) > thr

    # order-1 conditional screen: drop pairs explained by a common neighbor
    for i in range(d):
        for j in range(i + 1, d):
            if not adj[i, j]:
                continue
            for k in range(d):
                if k in (i, j) or not (adj[i, k] and adj[j, k]):
                    continue
                den = (1.0 - C[i, k] ** 2) * (1.0 - C[j, k] ** 2)
                if den <= 1e-10:
                    continue
                pc = (C[i, j] - C[i, k] * C[j, k]) / np.sqrt(den)
                if abs(pc) < thr:
                    adj[i, j] = adj[j, i] = False
                    break

    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if not adj[i, j]:
                continue
            dep_ij = _poly_resid_dependence(Z[:, i], Z[:, j])  # i -> j ?
            dep_ji = _poly_resid_dependence(Z[:, j], Z[:, i])  # j -> i ?
            if dep_ij <= dep_ji:
                B[j, i] = 1.0
            else:
                B[i, j] = 1.0
    return B
# =====================================================================

# =====================================================================
# EDITABLE: implement run_causal_discovery below
#
# Variant objective (confidence-gated orientation): measure how much
# identifying non-Gaussianity each variable actually carries, and let
# that measurement set the orientation machinery's aggressiveness.
# Pairs with enough signal use a residual higher-order-dependence
# contrast; pairs below the floor fall back to a cheap marginal
# heuristic (the CLT pushes effects toward Gaussian, so the more
# non-Gaussian variable is guessed to be the cause). Calibrating the
# floor from uncertainty rather than a constant, and replacing the
# fallback with something principled, is the headroom.
# =====================================================================
_GAUSS_LOGCOSH = 0.37457  # E[log cosh v] for v ~ N(0, 1)


def _nongauss_index(z: np.ndarray) -> float:
    """Bounded negentropy proxy; overflow-safe log-cosh contrast."""
    lc = np.logaddexp(z, -z) - 0.6931471805599453  # log(cosh(z))
    return float(abs(lc.mean() - _GAUSS_LOGCOSH))


def _hoc_dependence(cause: np.ndarray, resid: np.ndarray) -> float:
    """Higher-order dependence left between a regressor and its residual."""
    return float(abs(np.mean(cause ** 3 * resid)) + abs(np.mean(cause * resid ** 3)))


def run_causal_discovery(X: np.ndarray, corr_thresh: float = 0.3,
                         conf_floor: float = 0.02) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)

    ``conf_floor`` is the per-pair non-Gaussianity level below which
    the primary statistic is considered untrustworthy and the fallback
    takes over. The skeleton itself is a plain correlation threshold.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    Z /= np.where(sd > 0, sd, 1.0)

    ng = np.array([_nongauss_index(Z[:, i]) for i in range(d)])
    C = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(C, 0.0)

    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            r = float(C[i, j])
            if abs(r) <= corr_thresh:
                continue
            if min(ng[i], ng[j]) >= conf_floor:
                e_ji = Z[:, j] - r * Z[:, i]   # residual of j regressed on i
                e_ij = Z[:, i] - r * Z[:, j]
                i_causes = _hoc_dependence(Z[:, i], e_ji) < _hoc_dependence(Z[:, j], e_ij)
            else:
                i_causes = ng[i] >= ng[j]      # fallback: cause looks less Gaussianized
            if i_causes:
                B[j, i] = r
            else:
                B[i, j] = r
    return B
# =====================================================================

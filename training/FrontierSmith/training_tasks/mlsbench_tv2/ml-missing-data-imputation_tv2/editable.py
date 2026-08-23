# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant objective: an imputer whose derivation never assumes the
# holes fell at random. The missingness pattern is treated as data:
# fit() measures how each column's mask co-varies with every other
# column's observed values, and the fill anchors on per-column
# medians (which degrade more gracefully than means under selective
# observation). The measured mask-value dependence is currently
# diagnostic only -- promoting it into an explicit shift correction
# is the intended work.


def mask_value_dependence(X):
    """How each column's missingness relates to the other columns' values.

    Args:
        X: array (n_samples, n_features) with NaNs

    Returns:
        D: array (n_features, n_features); D[j, k] is the correlation
        between the missingness indicator of column j and the observed
        values of column k (0.0 where undefined). Large |D[j, k]| is
        evidence against a chance mask: whether j is missing carries
        information about the row's values.
    """
    n, d = X.shape
    D = np.zeros((d, d))
    nanmask = np.isnan(X)
    for j in range(d):
        mj = nanmask[:, j].astype(np.float64)
        if mj.std() == 0.0:
            continue
        for k in range(d):
            if k == j:
                continue
            obs = ~nanmask[:, k]
            if obs.sum() < 8 or mj[obs].std() == 0.0:
                continue
            vk = X[obs, k]
            if vk.std() == 0.0:
                continue
            c = np.corrcoef(mj[obs], vk)[0, 1]
            if np.isfinite(c):
                D[j, k] = c
    return D


class CustomImputer(BaseEstimator, TransformerMixin):
    """Missingness-mechanism-agnostic imputer (MNAR-aware scaffold).

    Must implement:
        fit(X) -> self              : learn imputation model from X (with NaNs)
        transform(X) -> X_imputed   : impute missing values in X

    Design targets for this variant:
    - No step may rely on the mask being independent of the data. Anchors
      are per-column medians; the matrix from `mask_value_dependence` is
      the evidence base for corrections.
    - `_shift(j)` is the correction hook: a signed adjustment added to
      column j's anchor for its missing cells, to be estimated from the
      dependence structure (placeholder: zero -- pure median fill).

    Args:
        random_state: seed for reproducibility.
        max_iter: budget for iterative correction passes (unused by the
            placeholder).
        shift_strength: scales `_shift` corrections; 0.0 disables them.
    """

    def __init__(self, random_state=42, max_iter=10, shift_strength=0.0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.shift_strength = shift_strength

    def fit(self, X, y=None):
        """Robust anchors + mask-value dependence evidence."""
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        self.anchor_ = np.where(np.isfinite(med), med, 0.0)
        self.dependence_ = mask_value_dependence(X)
        self.mask_rate_ = np.isnan(X).mean(axis=0)
        return self

    def _shift(self, j):
        """Signed anchor correction for column j's missing cells.

        Placeholder: 0.0. Intended: infer from self.dependence_ (and the
        rows' own observed values) whether column j's missing entries come
        from a shifted part of its distribution, and correct accordingly.
        """
        return 0.0

    def transform(self, X):
        """Median-anchored fill with (currently inert) shift corrections."""
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nanmask = np.isnan(X_imputed)
        for j in range(X_imputed.shape[1]):
            fill = self.anchor_[j] + self.shift_strength * self._shift(j)
            X_imputed[nanmask[:, j], j] = fill
        return X_imputed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

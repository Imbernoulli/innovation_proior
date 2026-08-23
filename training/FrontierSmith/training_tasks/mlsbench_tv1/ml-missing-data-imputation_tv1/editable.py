# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant objective: make reconstruction error (RMSE on the masked
# entries) and downstream utility move TOGETHER, with one fixed
# configuration across every dataset, and be judged by the weakest
# dataset. The scaffold exposes the adaptation channels this needs:
# per-column spread/missingness statistics and a budgeted refinement
# loop. The placeholder is a plain conditional-mean fill -- deliberately
# variance-collapsing, so the RMSE/downstream disagreement is visible.


def column_profile(X):
    """Per-column statistics of a NaN-bearing matrix.

    Args:
        X: array (n_samples, n_features) with NaNs

    Returns:
        dict with 'mean', 'std', 'missing_frac' arrays of shape
        (n_features,) -- the continuous statistics a single fixed
        configuration should adapt through (never dataset identity).
    """
    return {
        "mean": np.nanmean(X, axis=0),
        "std": np.nanstd(X, axis=0),
        "missing_frac": np.isnan(X).mean(axis=0),
    }


class CustomImputer(BaseEstimator, TransformerMixin):
    """Imputer judged on RMSE/downstream agreement at its weakest dataset.

    Must implement:
        fit(X) -> self              : learn imputation model from X (with NaNs)
        transform(X) -> X_imputed   : impute missing values in X

    Design targets for this variant:
    - ONE fixed configuration for all datasets (no shape/identity branching);
      adapt only through statistics like those in `column_profile`.
    - Completions should be accurate on the masked entries AND preserve the
      spread/dependence structure the downstream learner consumes. The
      `preserve_spread` knob and `_refine` hook are the intended levers.
    - Deterministic given random_state; always return finite values.

    Args:
        random_state: seed for reproducibility.
        max_iter: refinement budget (passes of `_refine` per transform).
        preserve_spread: float in [0, 1]; how strongly imputations should
            restore the observed per-column spread. 0.0 (default) is the
            pure conditional-mean placeholder, which collapses variance.
    """

    def __init__(self, random_state=42, max_iter=10, preserve_spread=0.0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.preserve_spread = preserve_spread

    def fit(self, X, y=None):
        """Learn imputation statistics from X (NaNs mark missing entries)."""
        profile = column_profile(X)
        self.statistics_ = profile["mean"]
        self.spread_ = profile["std"]
        self.missing_frac_ = profile["missing_frac"]
        self.n_refine_calls_ = 0  # budget counter for the refinement loop
        return self

    def _refine(self, X_imputed, missing_mask, iteration):
        """One refinement pass over the current completion.

        Args:
            X_imputed: (n_samples, n_features), current completion, no NaNs
            missing_mask: boolean mask of originally missing entries
            iteration: 0-based pass index (< self.max_iter)

        Returns:
            updated matrix, or None to stop early.

        Placeholder: returns None immediately -- the initial mean fill is
        never improved, which is exactly the weakness to attack (e.g.
        conditional re-prediction of masked cells, spread restoration
        scaled by `preserve_spread`, convergence checks against BOTH
        reconstruction and structure preservation).
        """
        return None

    def transform(self, X):
        """Impute missing values in X; returns a fully finite matrix."""
        X_imputed = X.copy()
        missing_mask = np.isnan(X_imputed)
        for j in range(X_imputed.shape[1]):
            X_imputed[missing_mask[:, j], j] = self.statistics_[j]
        for it in range(int(self.max_iter)):
            self.n_refine_calls_ += 1
            updated = self._refine(X_imputed, missing_mask, it)
            if updated is None:
                break
            X_imputed = updated
        return X_imputed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


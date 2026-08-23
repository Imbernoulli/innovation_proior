# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant objective: survive the stress regime where a typical row
# carries several holes at once. Structure provided: columns are
# completed in order of increasing missingness (reliable fills first,
# so they can support harder ones), heavily damaged rows are routed
# to an explicit degraded-row policy instead of a long inference
# chain, and a per-column chain hook exists for prediction-based
# refinement. The placeholder keeps every hook trivial (per-column
# means), so error compounding is currently avoided only by never
# chaining at all.


class CustomImputer(BaseEstimator, TransformerMixin):
    """Compounding-aware imputer for the many-holes-per-row regime.

    Must implement:
        fit(X) -> self              : learn imputation model from X (with NaNs)
        transform(X) -> X_imputed   : impute missing values in X

    Design targets for this variant:
    - Completion order is a first-class decision: `self.order_` ranks
      columns least-missing first, and `_chain_step` fills one column at
      a time so later columns may consult earlier completions. The
      contribution is a chain whose per-cell boldness reflects how much
      already-imputed (vs. genuinely observed) evidence it stands on.
    - Rows whose missing fraction exceeds `row_stress_threshold` are
      completed by `_degraded_row_fill` -- deliberately simple, because
      long chains on scant evidence compound errors.

    Args:
        random_state: seed for reproducibility.
        max_iter: cap on full chained passes over the columns.
        row_stress_threshold: missing fraction above which a row is
            treated as degraded (default 0.5).
    """

    def __init__(self, random_state=42, max_iter=10, row_stress_threshold=0.5):
        self.random_state = random_state
        self.max_iter = max_iter
        self.row_stress_threshold = row_stress_threshold

    def fit(self, X, y=None):
        """Column statistics + completion order (least-missing first)."""
        X = np.asarray(X, dtype=np.float64)
        mu = np.nanmean(X, axis=0)
        self.col_mean_ = np.where(np.isfinite(mu), mu, 0.0)
        self.col_missing_ = np.isnan(X).mean(axis=0)
        self.order_ = np.argsort(self.col_missing_, kind="stable")
        return self

    def _chain_step(self, X_partial, j, fill_rows):
        """Values for column j's missing cells given the partial completion.

        Args:
            X_partial: matrix in which columns earlier in self.order_ are
                already complete (their fills are available as evidence).
            j: index of the column being completed.
            fill_rows: boolean mask of rows needing a value in column j.

        Returns:
            array of length fill_rows.sum() with the values to write.

        Placeholder: the column mean -- it ignores the chain entirely,
        hence never compounds and never benefits.
        """
        return np.full(int(fill_rows.sum()), self.col_mean_[j])

    def _degraded_row_fill(self, X_imputed, nanmask, stressed_rows):
        """Policy for rows too damaged to support chained inference.

        Placeholder: column means (identical to the chain's placeholder);
        kept separate so the two regimes can diverge.
        """
        for j in range(X_imputed.shape[1]):
            rows = stressed_rows & nanmask[:, j]
            X_imputed[rows, j] = self.col_mean_[j]
        return X_imputed

    def transform(self, X):
        """Ordered, stress-aware completion; always returns finite values."""
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nanmask = np.isnan(X_imputed)
        row_frac = nanmask.mean(axis=1)
        stressed = row_frac > self.row_stress_threshold
        X_imputed = self._degraded_row_fill(X_imputed, nanmask, stressed)
        for j in self.order_:
            rows = nanmask[:, j] & ~stressed
            if rows.any():
                X_imputed[rows, j] = self._chain_step(X_imputed, j, rows)
        return X_imputed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant objective: boldness must be earned by evidence. fit()
# scores each column's evidential support (observed fraction times
# the strength of its best dependence on another column) and the
# completion interpolates between a conservative anchor (the column
# mean) and a model-based estimate as support grows. The placeholder
# pins the interpolation at the anchor end (full conservatism), so
# the confidence machinery is measured but not yet spent.


class CustomImputer(BaseEstimator, TransformerMixin):
    """Evidence-calibrated imputer: deviate from the anchor only with support.

    Must implement:
        fit(X) -> self              : learn imputation model from X (with NaNs)
        transform(X) -> X_imputed   : impute missing values in X

    Design targets for this variant:
    - self.evidence_[j] in [0, 1]: column j's support for model-based
      inference, computed as observed fraction times the strength of its
      best linear coupling to any other column.
    - `_estimate(X_anchor, j, rows)`: model-based candidate values for
      column j's missing cells, or None when no candidate is offered
      (placeholder: always None, so every cell stays at its anchor).
    - `conservatism` in [0, 1]: how much of the evidence gate is held
      back; 1.0 (default) means candidates are never trusted at all.
      The monotone evidence-to-boldness gate in transform() is the
      intended calibration surface.

    Args:
        random_state: seed for reproducibility.
        max_iter: budget for repeated estimate passes (unused by the
            placeholder).
        conservatism: float in [0, 1]; 1.0 = pure anchor fill.
    """

    def __init__(self, random_state=42, max_iter=10, conservatism=1.0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.conservatism = conservatism

    def fit(self, X, y=None):
        """Anchors plus a per-column evidence score in [0, 1]."""
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        mu = np.nanmean(X, axis=0)
        self.anchor_ = np.where(np.isfinite(mu), mu, 0.0)
        obs_frac = 1.0 - np.isnan(X).mean(axis=0)
        # Coupling strength: correlations on an anchor-filled copy.
        Xa = X.copy()
        nanmask = np.isnan(Xa)
        for j in range(d):
            Xa[nanmask[:, j], j] = self.anchor_[j]
        sd = Xa.std(axis=0)
        ok = sd > 1e-12
        C = np.zeros((d, d))
        if int(ok.sum()) >= 2:
            sub = np.corrcoef(Xa[:, ok], rowvar=False)
            sub = np.nan_to_num(sub, nan=0.0)
            np.fill_diagonal(sub, 0.0)
            C[np.ix_(ok, ok)] = sub
        coupling = np.abs(C).max(axis=1)
        self.evidence_ = np.clip(obs_frac * coupling, 0.0, 1.0)
        return self

    def _estimate(self, X_anchor, j, rows):
        """Model-based candidates for column j's missing cells, or None.

        Placeholder: None -- no candidate is ever offered. Intended: a
        dependence-based predictor whose output is trusted only through
        the evidence gate in transform().
        """
        return None

    def transform(self, X):
        """Anchor fill, then evidence-gated deviations where offered."""
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nanmask = np.isnan(X_imputed)
        for j in range(X_imputed.shape[1]):
            X_imputed[nanmask[:, j], j] = self.anchor_[j]
        for j in range(X_imputed.shape[1]):
            rows = nanmask[:, j]
            if not rows.any():
                continue
            candidate = self._estimate(X_imputed, j, rows)
            if candidate is None:
                continue
            gate = (1.0 - self.conservatism) * float(self.evidence_[j])
            X_imputed[rows, j] = (
                (1.0 - gate) * self.anchor_[j]
                + gate * np.asarray(candidate, dtype=np.float64)
            )
        return X_imputed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

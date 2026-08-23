# ================================================================
# EDITABLE -- agent modifies this section
# ================================================================
# Variant objective: respect what each column IS. fit() types every
# column from its observed values (small discrete support vs.
# continuous range), completions use a type-appropriate center (mode
# on the observed support vs. mean), and a projection hook snaps any
# refined estimate back onto a discrete column's support. Refinement
# itself is off in the placeholder: the value of the scaffold is that
# no imputed cell is off-support or out of scale for its column.


class CustomImputer(BaseEstimator, TransformerMixin):
    """Column-typing imputer: every fill must be plausible for its column.

    Must implement:
        fit(X) -> self              : learn imputation model from X (with NaNs)
        transform(X) -> X_imputed   : impute missing values in X

    Design targets for this variant:
    - Typing is inferred from observed values only (never hard-coded): a
      column whose observed values form a small set (at most
      discrete_max_card distinct values) is 'discrete'; anything else is
      'continuous'.
    - Discrete columns are filled on-support (placeholder: the mode);
      continuous columns are filled in-range (placeholder: the mean).
      `_snap` projects arbitrary estimates back onto a discrete column's
      support so any future model-based refinement stays type-faithful.
    - A wrong type call must degrade gracefully: labelling a continuous
      column 'discrete' merely restricts fills to observed values, which
      remains sane.

    Args:
        random_state: seed for reproducibility.
        max_iter: refinement budget (unused by the placeholder).
        discrete_max_card: max distinct observed values for a column to
            be treated as discrete (default 12).
    """

    def __init__(self, random_state=42, max_iter=10, discrete_max_card=12):
        self.random_state = random_state
        self.max_iter = max_iter
        self.discrete_max_card = discrete_max_card

    def fit(self, X, y=None):
        """Type each column and record its type-appropriate center."""
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        self.kind_ = []
        self.support_ = []
        self.center_ = np.zeros(d)
        for j in range(d):
            v = X[~np.isnan(X[:, j]), j]
            if v.size == 0:
                self.kind_.append("continuous")
                self.support_.append(None)
                self.center_[j] = 0.0
                continue
            uniq, counts = np.unique(v, return_counts=True)
            if uniq.size <= self.discrete_max_card:
                self.kind_.append("discrete")
                self.support_.append(uniq)
                self.center_[j] = float(uniq[np.argmax(counts)])  # mode
            else:
                self.kind_.append("continuous")
                self.support_.append(None)
                self.center_[j] = float(v.mean())
        return self

    def _snap(self, j, values):
        """Project estimates for column j onto its observed support.

        Discrete columns: each value is replaced by the nearest member of
        the observed support. Continuous columns: values pass through
        (range clipping is a candidate refinement).
        """
        if self.kind_[j] != "discrete":
            return values
        support = self.support_[j]
        vals = np.asarray(values, dtype=np.float64)
        idx = np.argmin(np.abs(vals[:, None] - support[None, :]), axis=1)
        return support[idx]

    def transform(self, X):
        """Type-faithful fill: mode-on-support or mean, per column kind."""
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nanmask = np.isnan(X_imputed)
        for j in range(X_imputed.shape[1]):
            rows = nanmask[:, j]
            if not rows.any():
                continue
            fills = np.full(int(rows.sum()), self.center_[j])
            X_imputed[rows, j] = self._snap(j, fills)
        return X_imputed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

class CalibrationMethod:
    """One rule for everyone: recalibration with zero per-group parameters.

    Variant objective: subgroup reliability must be earned by a single
    shared transform — nothing in the prediction path may key on a group
    id. Whatever protection the worst subgroup receives has to come from
    choosing the pooled mapping well, and the defence must show where
    per-group flexibility would genuinely have been unnecessary (or
    harmful) on the shifted tail.

    Structural hooks provided by this scaffold:
      - global Platt refit: a two-parameter logistic map (slope and
        intercept on the logit scale) fitted on all calibration pairs,
        with group ids deliberately discarded on entry.
      - ``slope_`` / ``intercept_``: the entire fitted state — the point
        of the constraint is that this is ALL the freedom available.
      - ``used_groups_``: stays False as a contract witness certifying
        that neither fit nor prediction consulted group membership.
    """

    def __init__(self):
        self.eps = 1e-6
        self.slope_ = 1.0
        self.intercept_ = 0.0
        self.used_groups_ = False
        self._model = None

    def fit(self, probs, labels, groups=None):
        # ``groups`` is intentionally ignored: the constraint of this
        # variant is a single global rule.
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self._model = None
        self.slope_ = 1.0
        self.intercept_ = 0.0
        if np.unique(labels).size == 2:
            p = np.clip(probs, self.eps, 1.0 - self.eps)
            z = (np.log(p) - np.log1p(-p)).reshape(-1, 1)
            model = LogisticRegression(C=1e6, max_iter=1000)
            model.fit(z, labels)
            self._model = model
            self.slope_ = float(model.coef_.ravel()[0])
            self.intercept_ = float(model.intercept_.ravel()[0])
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        if self._model is None:
            return p
        z = np.log(p) - np.log1p(-p)
        q = special.expit(self.slope_ * z + self.intercept_)
        return np.clip(q, self.eps, 1.0 - self.eps)

class CalibrationMethod:
    """Minimax recalibration: only the weakest subgroup's reliability counts.

    Variant objective: drive down the calibration error of whichever
    subgroup is worst on the shifted tail. Aggregate columns are
    guardrails, not goals — they must not collapse, but modelling effort
    belongs to the current arg-max of per-group ECE.

    Structural hooks provided by this scaffold:
      - ``group_ece_``: per-group calibration error measured at fit time —
        the map of where the maximum sits before any repair.
      - ``worst_group_``: the fitted arg-max; the placeholder repairs ONLY
        this group, with a logit-space bias that re-centres its mean
        confidence on its empirical positive rate.
      - ``group_bias_``: the applied per-group logit shifts (zero for all
        non-worst groups) — the lever a stronger minimax method broadens.
    """

    def __init__(self):
        self.eps = 1e-6
        self.group_ece_ = {}
        self.worst_group_ = None
        self.group_bias_ = {}

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.group_ece_ = {}
        self.worst_group_ = None
        self.group_bias_ = {}
        if groups is None:
            return self
        groups = np.asarray(groups).reshape(-1).astype(int)
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() < 5:
                continue
            self.group_ece_[int(g)] = expected_calibration_error(probs[mask], labels[mask])
        if not self.group_ece_:
            return self
        self.worst_group_ = int(max(self.group_ece_, key=self.group_ece_.get))
        mask = groups == self.worst_group_
        p = np.clip(probs[mask], self.eps, 1.0 - self.eps)
        rate = float(np.clip(labels[mask].mean(), self.eps, 1.0 - self.eps))
        mean_logit = float(np.mean(np.log(p) - np.log1p(-p)))
        target_logit = float(np.log(rate) - np.log1p(-rate))
        self.group_bias_ = {self.worst_group_: target_logit - mean_logit}
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        if groups is None or not self.group_bias_:
            return p
        groups = np.asarray(groups).reshape(-1).astype(int)
        logit = np.log(p) - np.log1p(-p)
        shift = np.array([self.group_bias_.get(int(g), 0.0) for g in groups])
        p = special.expit(logit + shift)
        return np.clip(p, self.eps, 1.0 - self.eps)

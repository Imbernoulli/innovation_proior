class CalibrationMethod:
    """Headcount-blind reliability: pooled corrections no group is too small for.

    Variant objective: the between-subgroup error gap is usually set by the
    smallest groups, because naive per-group fits are noisiest exactly
    where members are scarce. Every group — five members or five thousand
    — must end up with probabilities it can trust equally, so corrections
    borrow strength from the pooled data instead of leaning on thin
    per-group evidence.

    Structural hooks provided by this scaffold:
      - empirical-Bayes prevalence pooling: each group's logit offset is
        the gap between its own positive rate and the pooled rate, damped
        by ``n_g / (n_g + prior_strength)`` so tiny groups stay near the
        global map while large ones keep their own correction.
      - ``prior_strength``: the pseudo-count governing that damping — the
        single dial trading per-group fidelity against noise.
      - ``group_weight_`` / ``group_delta_``: fitted damping factors and
        offsets, the audit trail for the parity argument.
    """

    def __init__(self, prior_strength=64.0):
        self.eps = 1e-6
        self.prior_strength = float(prior_strength)
        self.pooled_rate_ = 0.5
        self.group_weight_ = {}
        self.group_delta_ = {}

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self.pooled_rate_ = float(np.clip(labels.mean(), self.eps, 1.0 - self.eps))
        self.group_weight_ = {}
        self.group_delta_ = {}
        if groups is None:
            return self
        groups = np.asarray(groups).reshape(-1).astype(int)
        pooled_logit = float(np.log(self.pooled_rate_) - np.log1p(-self.pooled_rate_))
        for g in np.unique(groups):
            mask = groups == g
            n_g = float(mask.sum())
            rate_g = float(np.clip(labels[mask].mean(), self.eps, 1.0 - self.eps))
            weight = n_g / (n_g + self.prior_strength)
            delta = weight * (float(np.log(rate_g) - np.log1p(-rate_g)) - pooled_logit)
            self.group_weight_[int(g)] = weight
            self.group_delta_[int(g)] = delta
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        if groups is None or not self.group_delta_:
            return p
        groups = np.asarray(groups).reshape(-1).astype(int)
        logit = np.log(p) - np.log1p(-p)
        # Unseen group ids fall back to the pooled map (zero offset).
        delta = np.array([self.group_delta_.get(int(g), 0.0) for g in groups])
        p = special.expit(logit + delta)
        return np.clip(p, self.eps, 1.0 - self.eps)

class CalibrationMethod:
    """Shift-robust subgroup calibration that does not pay in sharpness.

    Variant objective: keep worst-subgroup ECE and the between-subgroup gap
    low on the SHIFTED test tail while holding the Brier score — i.e.
    reliability must not be bought by flattening probabilities toward a
    base rate, nor by per-group curves the group's sample size cannot
    support.

    Structural hooks provided by this scaffold:
      - ``shrinkage``: float in [0, 1]; how strongly per-group corrections
        are pulled back toward the global mapping. 1.0 (default) disables
        per-group correction entirely, so the placeholder is a clipped
        identity map — its subgroup gap on the shifted tail is the
        weakness to attack.
      - ``group_support_`` / ``group_rate_``: per-group calibration sample
        counts and label rates; any support-weighted correction should be
        keyed on these, not on raw per-group curve fits.
      - ``_group_correction(g)``: additive logit-space correction for
        subgroup ``g`` (0.0 in the placeholder) — the intended lever.
    """

    def __init__(self, shrinkage=1.0):
        self.eps = 1e-6
        self.shrinkage = float(shrinkage)
        self._base_rate = 0.5
        self.group_support_ = {}
        self.group_rate_ = {}

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self._base_rate = float(np.clip(labels.mean(), self.eps, 1.0 - self.eps))
        self.group_support_ = {}
        self.group_rate_ = {}
        if groups is not None:
            groups = np.asarray(groups).reshape(-1).astype(int)
            for g in np.unique(groups):
                mask = groups == g
                self.group_support_[int(g)] = int(mask.sum())
                self.group_rate_[int(g)] = float(
                    np.clip(labels[mask].mean(), self.eps, 1.0 - self.eps)
                )
        return self

    def _group_correction(self, g):
        """Logit-space correction for subgroup ``g`` (placeholder: 0.0).

        A support-aware design scales this by how much calibration
        evidence group ``g`` has (see ``group_support_``), so small groups
        inherit the global mapping instead of their own noisy fit.
        """
        return 0.0

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        if groups is None or not self.group_support_ or self.shrinkage >= 1.0:
            return p
        logit = np.log(p) - np.log1p(-p)
        correction = np.array(
            [self._group_correction(int(g)) for g in np.asarray(groups).reshape(-1)]
        )
        logit = logit + (1.0 - self.shrinkage) * correction
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.clip(p, self.eps, 1.0 - self.eps)

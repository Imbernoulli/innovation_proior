class CalibrationMethod(BaseEstimator):
    """Sharpness-preserving recalibration under a worst-setting objective.

    Objective of this variant: ONE mapping (fixed hyperparameters, no
    per-setting switches) that lowers ECE without regressing Brier or NLL on
    any of the four classifier-dataset pairs — including a random forest that
    emits exact 0/1 vote ratios and an SVM already passed through a sigmoid —
    each fitted only on its modest calibration split.

    Interface (fixed):
        fit(probs, labels): probs (n,) binary positive-class probs or (n, C)
            multiclass rows summing to 1; labels (n,) integer classes.
        predict_proba(probs) -> calibrated probabilities, same shape as the
            input; multiclass rows must sum to 1.

    Structure provided as the intended adaptation channels:
      - self.base_rates_   : empirical class frequencies of the calibration
        set, learned in fit.
      - self.reliability_  : 10-bin (confidence, accuracy, count) summary of
        the calibration set — a cheap picture of WHERE the classifier is
        miscalibrated; the placeholder records but never uses it.
      - self.shrinkage / _shrink_lambda(): the placeholder mapping mixes the
        input with the base-rate prior, p <- (1-lam)*p + lam*pi, with a small
        constant lam. This floors the forest's exact 0/1 outputs (bounding
        NLL) but barely moves ECE — replacing this uniform shrinkage with a
        real, shape-aware, sample-size-aware correction is the task.
    """

    def __init__(self):
        self.is_binary = None
        self.base_rates_ = None
        self.reliability_ = None
        self.n_cal_ = 0
        # Fixed mixing weight toward the calibration base rates.
        self.shrinkage = 0.05

    def _shrink_lambda(self):
        """Mixing weight in [0, 1). Placeholder: constant, ignoring both the
        calibration-set size (self.n_cal_) and the reliability summary."""
        return self.shrinkage

    def _reliability_summary(self, probs, labels, n_bins=10):
        """10-bin (mean confidence, accuracy, count) table on the cal set."""
        if probs.ndim == 2:
            conf = np.max(probs, axis=1)
            correct = (np.argmax(probs, axis=1) == labels).astype(float)
        else:
            conf = np.where(probs >= 0.5, probs, 1.0 - probs)
            correct = ((probs >= 0.5).astype(int) == labels).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for i in range(n_bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if np.any(m):
                rows.append((float(conf[m].mean()), float(correct[m].mean()),
                             int(m.sum())))
            else:
                rows.append((0.5 * (edges[i] + edges[i + 1]), np.nan, 0))
        return rows

    def fit(self, probs, labels):
        """Learn the correction from the held-out calibration split."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        self.is_binary = probs.ndim == 1
        self.n_cal_ = labels.shape[0]
        if self.is_binary:
            self.base_rates_ = float(np.mean(labels == 1))
        else:
            c = probs.shape[1]
            counts = np.bincount(labels, minlength=c).astype(np.float64)
            self.base_rates_ = counts / max(1.0, counts.sum())
        self.reliability_ = self._reliability_summary(probs, labels)
        return self

    def predict_proba(self, probs):
        """Apply the mapping: shrink toward calibration base rates."""
        probs = np.asarray(probs, dtype=np.float64)
        lam = self._shrink_lambda()
        if self.is_binary:
            out = (1.0 - lam) * np.clip(probs, 0.0, 1.0) + lam * self.base_rates_
            return np.clip(out, 0.0, 1.0)
        out = (1.0 - lam) * probs + lam * self.base_rates_[None, :]
        out = np.clip(out, 1e-15, 1.0)
        return out / out.sum(axis=1, keepdims=True)

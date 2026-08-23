class CalibrationMethod(BaseEstimator):
    """Order-preserving recalibration: a strictly increasing map, nothing else.

    Objective of this variant: every repair must come from a monotone
    transform that can never swap two predictions or change an argmax —
    accuracy is invariant by construction. The placeholder fits the
    family's crudest member, a single temperature on log-probabilities,
    by minimising calibration-set NLL with a bounded scalar search.

    Interface (fixed):
        fit(probs, labels): probs (n,) binary positive-class probs or
            (n, C) multiclass rows summing to 1; labels (n,) ints.
        predict_proba(probs) -> same shape, valid probabilities.

    Mechanics and adaptation channels:
      - `temperature_`: the fitted scalar T > 0; binary probabilities map
        via expit(logit(p) / T), multiclass rows via softmax(log(p) / T) —
        both strictly increasing coordinate-wise.
      - `bounds`: search interval for T, fixed to [0.05, 20].
      - intended upgrades stay INSIDE the monotone family: positive-slope
        Platt maps, monotone splines with constrained coefficients, and
        compositions thereof — more curvature than one scalar can express,
        with the no-reordering guarantee intact.
    """

    def __init__(self):
        self.is_binary = None
        self.temperature_ = 1.0
        self.bounds = (0.05, 20.0)

    @staticmethod
    def _clip(p):
        return np.clip(p, 1e-12, 1.0 - 1e-12)

    def _binary_nll(self, t, z, y):
        q = self._clip(special.expit(z / t))
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    def _multi_nll(self, t, logp, y):
        s = special.softmax(logp / t, axis=1)
        return float(-np.mean(np.log(self._clip(s[np.arange(len(y)), y]))))

    def fit(self, probs, labels):
        """Fit the temperature by bounded scalar search on cal-set NLL."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        self.is_binary = probs.ndim == 1
        if self.is_binary:
            z = special.logit(self._clip(probs))
            y = (labels == 1).astype(float)
            res = optimize.minimize_scalar(
                lambda t: self._binary_nll(t, z, y),
                bounds=self.bounds, method="bounded",
            )
        else:
            logp = np.log(self._clip(probs))
            res = optimize.minimize_scalar(
                lambda t: self._multi_nll(t, logp, labels),
                bounds=self.bounds, method="bounded",
            )
        self.temperature_ = float(res.x)
        return self

    def predict_proba(self, probs):
        """Apply the strictly increasing temperature map."""
        probs = np.asarray(probs, dtype=np.float64)
        t = self.temperature_
        if self.is_binary:
            return special.expit(special.logit(self._clip(probs)) / t)
        s = special.softmax(np.log(self._clip(probs)) / t, axis=1)
        return s / s.sum(axis=1, keepdims=True)

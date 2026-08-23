class CalibrationMethod(BaseEstimator):
    """Tail-focused repair: recalibrate only the high-confidence region.

    Objective of this variant: occupancy-weighted ECE and unbounded NLL are
    both dominated by the crowded near-certain bins of an overconfident
    classifier, so the correction lives entirely above a confidence
    threshold `tau`; predictions below it pass through untouched.

    Interface (fixed):
        fit(probs, labels): probs (n,) binary positive-class probs or
            (n, C) multiclass rows summing to 1; labels (n,) ints.
        predict_proba(probs) -> same shape, valid probabilities.

    Mechanics and adaptation channels:
      - `tau` (0.85, fixed): boundary of the "tail" being repaired.
      - fit measures the empirical accuracy of tail predictions and remaps
        confidence linearly from [tau, 1] with a slope matched to that
        accuracy; the region below tau is the identity.
      - `tail_table_`: a per-bin (confidence, accuracy, count) picture of
        the tail, recorded at fit time but never consulted — replacing the
        single global slope with a bin-resolved, uncertainty-aware repair
        is the intended upgrade.
      - binary predictions are corrected on the confidence scale and mapped
        back to the predicted side; multiclass winners are corrected and
        losers rescaled to keep rows valid.
    """

    def __init__(self):
        self.is_binary = None
        self.tau = 0.85
        self.slope_ = 1.0
        self.tail_table_ = None

    @staticmethod
    def _conf_correct(probs, labels):
        if probs.ndim == 2:
            conf = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == labels).astype(float)
        else:
            conf = np.where(probs >= 0.5, probs, 1.0 - probs)
            correct = ((probs >= 0.5).astype(int) == labels).astype(float)
        return conf, correct

    def fit(self, probs, labels):
        """Measure tail reliability on the calibration split."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        self.is_binary = probs.ndim == 1
        conf, correct = self._conf_correct(probs, labels)
        tail = conf > self.tau
        acc = float(correct[tail].mean()) if np.any(tail) else float(correct.mean())
        self.slope_ = float(np.clip((acc - self.tau) / (1.0 - self.tau), 0.05, 1.5))
        edges = np.linspace(self.tau, 1.0, 6)
        rows = []
        for i in range(5):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            mid = 0.5 * (edges[i] + edges[i + 1])
            rows.append((
                float(conf[m].mean()) if np.any(m) else mid,
                float(correct[m].mean()) if np.any(m) else np.nan,
                int(m.sum()),
            ))
        self.tail_table_ = rows
        return self

    def _remap_conf(self, conf):
        out = conf.copy()
        m = conf > self.tau
        out[m] = self.tau + (conf[m] - self.tau) * self.slope_
        return np.clip(out, 0.0, 1.0 - 1e-9)

    def predict_proba(self, probs):
        """Apply the tail remap; identity below tau."""
        probs = np.asarray(probs, dtype=np.float64)
        if self.is_binary:
            conf = np.where(probs >= 0.5, probs, 1.0 - probs)
            newc = self._remap_conf(conf)
            return np.where(probs >= 0.5, newc, 1.0 - newc)
        n, C = probs.shape
        conf = probs.max(axis=1)
        winners = probs.argmax(axis=1)
        newc = np.clip(self._remap_conf(conf), 1e-12, 1.0 - 1e-12)
        rest = np.maximum(1.0 - conf, 0.0)
        scale = np.where(rest > 1e-9, (1.0 - newc) / np.maximum(rest, 1e-9), 0.0)
        out = probs * scale[:, None]
        out[np.arange(n), winners] = newc
        deg = rest <= 1e-9
        if np.any(deg) and C > 1:
            out[deg] = ((1.0 - newc[deg]) / (C - 1))[:, None]
            out[deg, winners[deg]] = newc[deg]
        out = np.clip(out, 1e-15, 1.0)
        return out / out.sum(axis=1, keepdims=True)

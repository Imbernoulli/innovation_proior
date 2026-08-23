class CalibrationMethod(BaseEstimator):
    """Evidence-priced binning: capacity scales with the calibration split.

    Objective of this variant: one fixed method is fitted on splits ranging
    from ~114 points (breast cancer) to ~4000 (the image sets). The
    baseline is equal-frequency binning whose two capacity knobs are tied
    to n: the bin count follows a square-root rule and every bin accuracy
    is shrunk toward the global accuracy under a fixed pseudo-count. Both
    rules are placeholders for a principled evidence-pricing scheme
    (stability-driven bin counts, hierarchical priors, capacity chosen by
    within-split validation).

    Interface (fixed):
        fit(probs, labels): probs (n,) binary positive-class probs or
            (n, C) multiclass rows summing to 1; labels (n,) ints.
        predict_proba(probs) -> same shape, valid probabilities.

    Mechanics:
      - binary: bins the positive-class probability directly.
      - multiclass: bins the winner confidence, maps it through the learned
        table, and rescales the losing classes so rows still sum to 1
        (rows whose winner held all the mass spread the remainder evenly).
      - `_n_bins(n)`: sqrt-rule bin count, clipped to [4, 15].
      - `prior_strength`: pseudo-count (8.0) pulling small bins toward the
        split's global accuracy; an empty bin falls back to it entirely.
    """

    def __init__(self):
        self.is_binary = None
        self.prior_strength = 8.0
        self.edges_ = None
        self.values_ = None
        self.global_ = 0.5

    @staticmethod
    def _n_bins(n):
        return int(np.clip(int(np.sqrt(n) / 2.0), 4, 15))

    def _fit_table(self, conf, target):
        n = conf.shape[0]
        qs = np.quantile(conf, np.linspace(0.0, 1.0, self._n_bins(n) + 1))
        self.edges_ = np.unique(qs)[1:-1]  # interior cut points
        nb = len(self.edges_) + 1
        idx = np.searchsorted(self.edges_, conf, side="right")
        g = float(target.mean())
        m0 = self.prior_strength
        vals = np.empty(nb, dtype=np.float64)
        for i in range(nb):
            sel = idx == i
            vals[i] = (target[sel].sum() + m0 * g) / (sel.sum() + m0)
        self.values_, self.global_ = vals, g

    def _apply_table(self, conf):
        idx = np.searchsorted(self.edges_, conf, side="right")
        return self.values_[np.clip(idx, 0, len(self.values_) - 1)]

    def fit(self, probs, labels):
        """Learn the shrunken equal-frequency table from the cal split."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        self.is_binary = probs.ndim == 1
        if self.is_binary:
            self._fit_table(probs, (labels == 1).astype(float))
        else:
            conf = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == labels).astype(float)
            self._fit_table(conf, correct)
        return self

    def predict_proba(self, probs):
        """Map through the table; keep multiclass rows valid."""
        probs = np.asarray(probs, dtype=np.float64)
        if self.is_binary:
            return np.clip(self._apply_table(probs), 0.0, 1.0)
        n, C = probs.shape
        conf = probs.max(axis=1)
        winners = probs.argmax(axis=1)
        newc = np.clip(self._apply_table(conf), 1e-12, 1.0 - 1e-12)
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

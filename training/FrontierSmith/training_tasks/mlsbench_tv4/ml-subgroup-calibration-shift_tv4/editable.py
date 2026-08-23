class CalibrationMethod:
    """Extrapolation-safe recalibration for a test region the fit never saw.

    Variant objective: the evaluation queries the mapping on the far tail
    of a domain score, so behaviour OUTSIDE the fitted score range decides
    the outcome. Low-capacity monotone corrections that extend predictably
    beat flexible curves that excel where calibration data lives and turn
    unstable where it does not; degradation as the shift grows should be
    gradual, never a cliff.

    Structural hooks provided by this scaffold:
      - a single bounded temperature: logits are divided by ``T`` fitted
        by NLL on the calibration split — the lowest-capacity correction
        that still moves ECE, with smooth extrapolation by construction.
      - ``temperature_``: the fitted value, kept inside ``t_bounds``.
      - shift probe: ``fit`` refits the temperature on the lower half of
        the calibration score range and stores ``probe_gap_`` =
        ``|T_low - T_full|``, a cheap reading of shift sensitivity.
    """

    def __init__(self, t_bounds=(0.25, 4.0)):
        self.eps = 1e-6
        self.t_bounds = (float(t_bounds[0]), float(t_bounds[1]))
        self.temperature_ = 1.0
        self.probe_gap_ = 0.0

    def _fit_temperature(self, logit, labels):
        def nll(t):
            q = np.clip(special.expit(logit / t), self.eps, 1.0 - self.eps)
            return -float(np.mean(labels * np.log(q) + (1 - labels) * np.log(1.0 - q)))

        res = optimize.minimize_scalar(nll, bounds=self.t_bounds, method="bounded")
        return float(np.clip(res.x, self.t_bounds[0], self.t_bounds[1]))

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        logit = np.log(p) - np.log1p(-p)
        self.temperature_ = self._fit_temperature(logit, labels)
        # Shift probe: refit on the lower half of the score range only and
        # record how far the temperature moves.
        low = p <= float(np.median(p))
        if low.sum() >= 10 and np.unique(labels[low]).size == 2:
            t_low = self._fit_temperature(logit[low], labels[low])
            self.probe_gap_ = float(abs(t_low - self.temperature_))
        else:
            self.probe_gap_ = 0.0
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        logit = np.log(p) - np.log1p(-p)
        q = special.expit(logit / self.temperature_)
        return np.clip(q, self.eps, 1.0 - self.eps)

class CalibrationMethod(BaseEstimator):
    """Geometry-agnostic recalibration across heterogeneous classifier outputs.

    Objective of this variant: the same fixed method must digest vote
    ratios containing exact 0/1 mass, overconfident softmax outputs,
    staged boosting scores, and margins already squashed by an upstream
    sigmoid. The placeholder makes three geometry-neutral moves: clip away
    from the boundary so every input has a finite logit, correct each
    class independently with a two-parameter logistic in that logit
    domain, and renormalise.

    Interface (fixed):
        fit(probs, labels): probs (n,) binary positive-class probs or
            (n, C) multiclass rows summing to 1; labels (n,) ints.
        predict_proba(probs) -> same shape, valid probabilities.

    Mechanics and adaptation channels:
      - `eps`: boundary clip (1e-6) — the only defence against infinite
        logits from exact 0/1 vote ratios; whether boundary mass deserves
        different treatment from interior mass is an open design point.
      - `models_`: one LogisticRegression per class fitted one-vs-rest on
        that class's clipped logit (a degenerate class falls back to the
        identity). Per-class independence spends parameters freely — when
        classes should instead share strength (ten-way vs binary, small vs
        large splits) is the trade-off to interrogate.
      - `signature_`: a geometry diagnostic (boundary-mass fraction, mean
        winner confidence) recorded at fit time and never read.
    """

    def __init__(self):
        self.is_binary = None
        self.eps = 1e-6
        self.models_ = None
        self.signature_ = None

    def _logit(self, p):
        p = np.clip(np.asarray(p, dtype=np.float64), self.eps, 1.0 - self.eps)
        return np.log(p / (1.0 - p))

    def _fit_one(self, p, y):
        """Two-parameter logistic correction for one class; None if degenerate."""
        if y.min() == y.max():
            return None
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(self._logit(p).reshape(-1, 1), y)
        return m

    def _apply_one(self, model, p):
        if model is None:
            return np.clip(p, self.eps, 1.0 - self.eps)
        q = model.predict_proba(self._logit(p).reshape(-1, 1))
        return q[:, list(model.classes_).index(1)]

    def fit(self, probs, labels):
        """Fit per-class logit-domain corrections on the calibration split."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        self.is_binary = probs.ndim == 1
        flat = probs.ravel()
        winner = probs if self.is_binary else probs.max(axis=1)
        winner = np.maximum(winner, 1.0 - winner) if self.is_binary else winner
        self.signature_ = {
            "boundary_mass": float(
                np.mean((flat <= self.eps) | (flat >= 1.0 - self.eps))
            ),
            "mean_winner_conf": float(np.mean(winner)),
        }
        if self.is_binary:
            self.models_ = [self._fit_one(probs, (labels == 1).astype(int))]
        else:
            self.models_ = [
                self._fit_one(probs[:, j], (labels == j).astype(int))
                for j in range(probs.shape[1])
            ]
        return self

    def predict_proba(self, probs):
        """Apply per-class corrections; renormalise multiclass rows."""
        probs = np.asarray(probs, dtype=np.float64)
        if self.is_binary:
            return np.clip(self._apply_one(self.models_[0], probs), 0.0, 1.0)
        cols = [
            self._apply_one(self.models_[j], probs[:, j])
            for j in range(probs.shape[1])
        ]
        out = np.clip(np.column_stack(cols), 1e-15, None)
        return out / out.sum(axis=1, keepdims=True)

class BoostingStrategy:
    """Trust-aware boosting: bounded influence for possibly-corrupt labels.

    Variant premise: some training labels may be wrong, and boosting's
    classical response (escalating focus on unfittable points) is exactly
    backwards. Every channel through which one sample acts is therefore
    bounded: residual targets are clipped at a robust scale, coefficients
    are constant, and a consecutive-miss tracker exists to tell
    hard-but-genuine from likely-corrupt.

    Interface (fixed): init_weights, compute_targets, compute_learner_weight,
    update_weights -- called by the fixed training loop each round.

    Provided structure:
      - self.spread_: inter-quartile-range scale of the raw residuals,
        refreshed each regression round.
      - self.clip_k: targets are clipped to +/- clip_k * spread_
        (bounded influence, always on for regression; the logistic
        gradient used for classification is bounded by construction).
      - self.streak_: consecutive rounds each sample has disagreed with
        the ensemble (negative margin / residual beyond the clip).
      - self.patience, self.suspect_discount: weight multiplier applied
        once a streak reaches patience. Discount is 1.0, i.e. suspicion
        is recorded but never acted on -- deciding when and how hard to
        distrust is the intended contribution.
    """

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self.spread_ = 1.0
        self.clip_k = 3.0
        self.streak_ = None
        self.patience = 15
        self.suspect_discount = 1.0  # placeholder: suspicion changes nothing

    def init_weights(self, n_samples):
        """Uniform start; disagreement streaks start at zero."""
        self.streak_ = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Bounded pseudo-targets; refresh the robust spread and streaks."""
        if self.task_type == "classification":
            y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
            margin = np.clip(y_signed * current_predictions, -30.0, 30.0)
            r = y_signed / (1.0 + np.exp(margin))  # |r| <= 1 by construction
            disagree = margin < 0.0
        else:
            raw = np.asarray(y, dtype=np.float64) - current_predictions
            q75, q25 = np.percentile(raw, [75.0, 25.0])
            self.spread_ = float(q75 - q25) + 1e-8
            lim = self.clip_k * self.spread_
            r = np.clip(raw, -lim, lim)
            disagree = np.abs(raw) > lim
        if self.streak_ is None or len(self.streak_) != len(r):
            self.streak_ = np.zeros(len(r))
        self.streak_ = np.where(disagree, self.streak_ + 1.0, 0.0)
        if np.array_equal(r, r.astype(int)):
            r = r + 1e-6  # stay on the fixed loop's regression-tree path
        return r

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        """Constant unit coefficient: no round buys extra leverage."""
        return 1.0

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        """Discount samples whose disagreement streak exceeds patience."""
        n = len(sample_weights)
        w = np.ones(n) / n
        suspect = self.streak_ >= self.patience
        w = np.where(suspect, w * self.suspect_discount, w)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            return np.ones(n) / n
        return w / s

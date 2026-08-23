class BoostingStrategy:
    """Hard-minority boosting: emphasis must be earned, and it is capped.

    Variant premise: held-out numbers are decided by a minority of
    persistently difficult samples. This strategy accumulates difficulty
    EVIDENCE across rounds (an exponential moving average, not a single
    round's verdict) and converts it into sample weight on an explicit
    schedule, with a hard ceiling on any one sample's share so emphasis
    can never become capture.

    Interface (fixed): init_weights, compute_targets, compute_learner_weight,
    update_weights -- called by the fixed training loop each round.

    Provided structure:
      - self.hardness_: EMA of per-round difficulty (miss indicator for
        classification, normalised |residual| for regression) -- the
        persistence signal emphasis should be based on.
      - self.mix: fraction of weight allocated by difficulty vs. kept
        uniform. 0.3 is a mild, safe schedule; making it adaptive is the
        intended contribution.
      - self.share_cap: ceiling on one sample's fraction of total weight,
        enforced after every update.
    Classification runs error-weighted alphas on discrete targets;
    regression runs residual fitting with unit alphas.
    """

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self.hardness_ = None
        self.ema = 0.9
        self.mix = 0.3
        self.share_cap = 0.05

    def init_weights(self, n_samples):
        """Uniform start; difficulty evidence starts at zero."""
        self.hardness_ = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Discrete labels for classification; residuals for regression."""
        if self.task_type == "classification":
            return np.asarray(y)
        r = np.asarray(y, dtype=np.float64) - current_predictions
        if np.array_equal(r, r.astype(int)):
            r = r + 1e-6  # keep the fixed loop on regression trees
        return r

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        """Weighted-error alpha (classification); unit alpha (regression)."""
        if self.task_type != "classification":
            return 1.0
        miss = (learner.predict(X) != np.asarray(y)).astype(np.float64)
        self._last_miss = miss
        denom = max(float(np.sum(sample_weights)), 1e-12)
        err = float(np.sum(sample_weights * miss)) / denom
        err = min(max(err, 1e-6), 1.0 - 1e-6)
        alpha = 0.5 * np.log((1.0 - err) / err)
        return float(np.clip(alpha, -4.0, 4.0))

    def _cap_shares(self, w):
        """Enforce the per-sample share ceiling, then renormalise."""
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        w = np.minimum(w, self.share_cap)
        return w / w.sum()

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        """Fold this round's difficulty into the EMA; mix with uniform; cap."""
        if self.task_type == "classification":
            difficulty = getattr(self, "_last_miss",
                                 np.zeros_like(sample_weights))
        else:
            difficulty = np.abs(np.asarray(pseudo_targets, dtype=np.float64))
        d = difficulty / (float(difficulty.max()) + 1e-12)
        self.hardness_ = self.ema * self.hardness_ + (1.0 - self.ema) * d
        n = len(sample_weights)
        h = self.hardness_ / max(float(self.hardness_.sum()), 1e-12)
        w = (1.0 - self.mix) / n + self.mix * h
        return self._cap_shares(w)

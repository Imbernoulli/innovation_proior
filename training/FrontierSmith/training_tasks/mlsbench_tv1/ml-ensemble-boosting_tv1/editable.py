class BoostingStrategy:
    """Late-round-robust boosting: keep rounds 100-200 productive.

    Objective of this variant: a single strategy instance that serves both a
    near-separable classification task and noisy regression tasks, designed
    around the LATE rounds — where residuals are mostly noise/outliers and
    naive reweighting or residual-chasing overfits. The reweighting, damping,
    and scheduling machinery must be shared across task types; only the loss
    whose gradient is followed may differ.

    Interface (fixed): init_weights, compute_targets, compute_learner_weight,
    update_weights — called by the fixed training loop each round.

    Structure provided as the intended adaptation channels:
      - phase(round_idx): fraction of the run completed, in (0, 1] — a
        continuous schedule variable (the placeholder never uses it).
      - self.scale_: running robust residual scale (median absolute deviation),
        refreshed every round in compute_targets; the natural unit for
        deciding which residuals are "signal" and which are outliers.
      - _influence(residuals): per-sample influence in [0, 1] given the
        current residuals and scale. Placeholder: all ones (no robustness).
        Wiring this into the pseudo-targets and/or the sample weights is the
        intended contribution.
      - self.damping: constant conservative learner coefficient (0.5). A
        placeholder for a real, phase- and scale-aware step-size rule.
    """

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self.scale_ = 1.0
        self.damping = 0.5

    def phase(self, round_idx):
        """Progress through the run, in (0, 1]."""
        return (round_idx + 1) / float(max(1, self.n_rounds))

    def _influence(self, residuals):
        """Per-sample influence weights in [0, 1] (robustness hook).

        Placeholder: every sample counts fully, regardless of how far its
        residual sits from self.scale_.
        """
        return np.ones_like(np.asarray(residuals, dtype=np.float64))

    def init_weights(self, n_samples):
        """Uniform initial sample weights (sum to 1)."""
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Negative-gradient pseudo-targets; refreshes the residual scale.

        Classification: logistic-loss gradient on signed labels (bounded, so
        late rounds cannot chase unbounded margins). Regression: plain
        residuals. Both share the scale statistic below.
        """
        if self.task_type == "classification":
            y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
            margin = y_signed * current_predictions
            r = y_signed / (1.0 + np.exp(np.clip(margin, -30.0, 30.0)))
        else:
            r = np.asarray(y, dtype=np.float64) - current_predictions
        # Running robust scale of the residual distribution (MAD).
        self.scale_ = float(np.median(np.abs(r - np.median(r)))) + 1e-8
        # Guard: keep targets off the integer lattice so the fixed loop's
        # continuous/discrete detection always selects regression trees.
        if np.array_equal(r, r.astype(int)):
            r = r + 1e-6
        return r

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        """Constant damped coefficient (placeholder for a real step rule)."""
        return self.damping

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        """Influence-shaped sample weights (placeholder: uniform)."""
        w = self._influence(pseudo_targets)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(sample_weights) / len(sample_weights)
        return w / s

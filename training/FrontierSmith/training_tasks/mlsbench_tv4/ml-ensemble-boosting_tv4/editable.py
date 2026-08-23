class BoostingStrategy:
    """Anytime boosting: front-load quality, decelerate into consolidation.

    Variant premise: the run should be interruptible. Early rounds take
    the largest steps; a pacing schedule then decays the coefficient so
    late rounds refine without destabilizing. Per-round overhead is
    deliberately near zero (no reweighting machinery at all) -- budget
    discipline applies to the strategy's own compute too.

    Interface (fixed): init_weights, compute_targets, compute_learner_weight,
    update_weights -- called by the fixed training loop each round.

    Provided structure:
      - self.budget: round count by which quality should essentially be
        in place (n_rounds // 5, at least 8).
      - _pace(round_idx): step multiplier. Implemented: harmonic decay
        from self.boost0 inside the budget, then geometric fade at
        self.fade per round beyond it. Making the pace react to observed
        progress instead of a fixed clock is the intended contribution.
      - Targets are residuals in a common form for both task types
        (signed labels for classification), so pacing is the only moving
        part.
    """

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self.budget = max(8, self.n_rounds // 5)
        self.boost0 = 3.0   # initial step multiplier (effective step ~0.3)
        self.fade = 0.97    # per-round decay beyond the budget point

    def init_weights(self, n_samples):
        """Uniform weights; this variant never reweights samples."""
        return np.ones(n_samples) / n_samples

    def _pace(self, round_idx):
        """Front-loaded step multiplier for round `round_idx`."""
        if round_idx < self.budget:
            return self.boost0 / (1.0 + 3.0 * round_idx / float(self.budget))
        over = round_idx - self.budget
        return (self.boost0 / 4.0) * (self.fade ** over)

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Residuals of the running score against (signed) targets."""
        t = np.asarray(y, dtype=np.float64)
        if self.task_type == "classification":
            t = 2.0 * t - 1.0
        r = t - current_predictions
        if np.array_equal(r, r.astype(int)):
            r = r + 1e-6  # force the fixed loop's regression-tree path
        return r

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        """Alpha follows the pacing schedule (scaled by lr in the loop)."""
        return float(self._pace(round_idx))

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        """Keep the distribution untouched: zero bookkeeping by design."""
        return sample_weights

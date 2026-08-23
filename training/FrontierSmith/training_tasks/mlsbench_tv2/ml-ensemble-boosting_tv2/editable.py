class BoostingStrategy:
    """Capacity-starved boosting: the sequence, not the tree, carries skill.

    Variant premise: a depth-3 tree can express only a few axis-aligned
    cells, so no single round can be strong and all improvement must come
    from how rounds complement one another. The scaffold tracks, per
    sample, how much attention the run has already spent on each point,
    and exposes that record as the steering channel for pointing the next
    tree at territory the existing ensemble ignores.

    Interface (fixed): init_weights, compute_targets, compute_learner_weight,
    update_weights -- called by the fixed training loop each round.

    Provided structure:
      - self.attention_: running per-sample accumulation of past sample
        weight (how much focus each point has received). Refreshed in
        update_weights; nothing consumes it yet.
      - _novelty(attention): per-sample multipliers meant to favour
        under-served samples. Placeholder: uniform, i.e. successive trees
        are steered nowhere -- wiring this in is the point of the variant.
      - Coefficients are constant 1.0: the variant's leverage is WHERE
        trees look, never how hard they push.
    """

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self.attention_ = None

    def init_weights(self, n_samples):
        """Uniform start; the attention record starts empty."""
        self.attention_ = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def _novelty(self, attention):
        """Multipliers favouring samples the run has under-served.

        Placeholder: all ones (no steering).
        """
        return np.ones_like(attention)

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Plain negative-gradient targets (logistic / squared loss)."""
        if self.task_type == "classification":
            y_signed = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
            margin = np.clip(y_signed * current_predictions, -30.0, 30.0)
            r = y_signed / (1.0 + np.exp(margin))
        else:
            r = np.asarray(y, dtype=np.float64) - current_predictions
        if np.array_equal(r, r.astype(int)):
            r = r + 1e-6  # keep the fixed loop on regression trees
        return r

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        """Constant unit coefficient; shrinkage comes from the loop's lr."""
        return 1.0

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        """Record attention spent, then apply the novelty multipliers."""
        self.attention_ = self.attention_ + sample_weights
        w = sample_weights * self._novelty(self.attention_)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(sample_weights) / len(sample_weights)
        return w / s

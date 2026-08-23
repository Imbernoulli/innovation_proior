class CustomSampling(Strategy):
    """Cold-start-aware batch acquisition: coverage first, uncertainty once earned.

    Objective of this variant: maximize the area under the accuracy-vs-labels
    curve by making the EARLY rounds strong. While the model is untrustworthy
    the batch should come from label-free coverage of the pool; as evidence
    accumulates the batch should shift to model-driven selection. The handover
    must be driven by a measured statistic (see `_model_trust`), not by a
    hard-coded round schedule, and the same constants must serve all datasets.

    Interface (fixed): query(n) -> np.ndarray of n indices into self.X drawn
    from the currently unlabeled pool. Helper methods may be added freely.

    Structure provided as the intended adaptation channels:
      - self.round_        : number of query() calls made so far.
      - self._model_trust(): trust in the current model, in [0, 1]. The
        placeholder uses only labeled-set size; replace with a statistic that
        actually measures reliability (e.g. MC-dropout committee agreement via
        self.predict_prob_dropout_split, or round-to-round confidence
        stability).
      - coverage_floor     : fraction of each batch reserved for
        model-independent coverage at zero trust; decays as trust grows.
        The placeholder's "coverage" is uniform random — replace with real
        coverage (density / diversity / representativeness).
    """

    # Fraction of the batch chosen model-independently when trust == 0.
    coverage_floor = 0.9

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.round_ = 0

    def _model_trust(self):
        """Return trust in the current model as a float in [0, 1].

        Placeholder: labeled fraction relative to 10% of the pool. This is a
        deliberately crude proxy — it ignores what the model actually knows.
        """
        n_lb = float(np.sum(self.idxs_lb))
        return min(1.0, n_lb / max(1.0, 0.1 * self.n_pool))

    def query(self, n):
        """Select n unlabeled samples: a trust-weighted mix of coverage and
        least-confidence exploitation.
        """
        self.round_ += 1
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        trust = self._model_trust()
        mix = (1.0 - self.coverage_floor) + self.coverage_floor * trust
        n_exploit = int(round(mix * n))
        n_cover = n - n_exploit

        order = np.random.permutation(len(idxs_unlabeled))
        chosen = list(idxs_unlabeled[order[:n_cover]])
        rest = idxs_unlabeled[order[n_cover:]]
        if n_exploit > 0 and len(rest) > 0:
            probs = self.predict_prob(self.X[rest], np.asarray(self.Y)[rest])
            conf = probs.max(1)[0].numpy()
            chosen.extend(rest[np.argsort(conf)[:n_exploit]])
        return np.asarray(chosen[:n])

# ================================================================
# END EDITABLE REGION
# ================================================================

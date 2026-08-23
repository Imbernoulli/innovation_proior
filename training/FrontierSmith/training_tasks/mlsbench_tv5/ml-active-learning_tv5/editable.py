class CustomSampling(Strategy):
    """Portfolio acquisition: several experts share each batch.

    Variant objective: no-regret behavior across the whole curve. Each
    round's batch is split among qualitatively different experts --
    least-confidence, top-two-margin, predictive-entropy, and uniform --
    so no single heuristic's dead zone owns the round. The placeholder
    uses FIXED slot weights; the intended contribution is an adaptive
    reweighting driven by in-run evidence (prediction shift after each
    expert's purchases, redundancy of an expert's nominations with the
    labeled set), plus a principled treatment of overlapping
    nominations. Weights must be learned per run, never per dataset by
    hand.

    Interface (fixed): query(n) -> np.ndarray of n indices into self.X
    drawn from the currently unlabeled pool.
    """

    # Fixed slot allocation: (least-confidence, margin, entropy, uniform).
    slot_weights = (0.35, 0.25, 0.2, 0.2)

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.rng_ = np.random.RandomState(0)

    def query(self, n):
        """Fill expert quotas in turn from a shared candidate pool so
        overlapping nominations never charge the budget twice."""
        n = int(n)
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(
            self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled]
        ).numpy()

        srt = np.sort(probs, axis=1)
        scores = [
            1.0 - probs.max(axis=1),                       # least confidence
            -(srt[:, -1] - srt[:, -2]),                    # small margin
            -(probs * np.log(probs + 1e-12)).sum(axis=1),  # entropy
        ]

        quotas = [int(w * n) for w in self.slot_weights]
        quotas[-1] += n - sum(quotas)                      # remainder -> uniform

        taken = np.zeros(len(idxs_unlabeled), dtype=bool)
        picks = []
        for expert, q in enumerate(quotas[:-1]):
            order = np.argsort(-scores[expert])
            for k in order:
                if q <= 0:
                    break
                if not taken[k]:
                    taken[k] = True
                    picks.append(k)
                    q -= 1
        free = np.nonzero(~taken)[0]                       # uniform expert
        n_uni = min(quotas[-1], len(free))
        if n_uni > 0:
            picks.extend(self.rng_.choice(free, size=n_uni, replace=False))
        return idxs_unlabeled[np.asarray(picks[:n], dtype=int)]

# ================================================================
# END EDITABLE REGION
# ================================================================

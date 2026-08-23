class CustomSampling(Strategy):
    """Noise-hardened acquisition: informative middle, not the ambiguity tail.

    Variant objective: buy examples that are unresolved yet corroborable,
    on the premise that the extreme-ambiguity tail is where oracle noise
    and pathological instances concentrate. The placeholder ranks the
    unlabeled pool by top-two margin, then EXCLUDES the most ambiguous
    ``tail_frac`` of candidates and fills the batch from the smallest
    margins that survive the cut. The cut is crude by design: a real
    reliability signal (neighborhood label agreement, embedding density,
    committee variance) should replace the fixed quantile, and the
    discount should be soft rather than a hard exclusion. Same constants
    on all three datasets.

    Interface (fixed): query(n) -> np.ndarray of n indices into self.X
    drawn from the currently unlabeled pool.
    """

    # Fraction of the most-ambiguous candidates treated as untrustworthy.
    tail_frac = 0.05

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        """Band-pass margin selection: skip the pathological tail, then
        take the most informative of what remains."""
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(
            self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled]
        ).numpy()

        part = np.sort(probs, axis=1)
        margin = part[:, -1] - part[:, -2]      # small = ambiguous

        order = np.argsort(margin)              # most ambiguous first
        n_skip = int(self.tail_frac * len(order))
        kept = order[n_skip:]                   # drop the untrusted tail
        picks = kept[:int(n)]
        if len(picks) < int(n):                 # tiny pools: refill from tail
            picks = np.concatenate([picks, order[:n_skip]])[:int(n)]
        return idxs_unlabeled[picks]

# ================================================================
# END EDITABLE REGION
# ================================================================

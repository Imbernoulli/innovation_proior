class CustomSampling(Strategy):
    """Batch-diversity discipline: the batch is a set, not a top-k cut.

    Variant objective: eliminate within-batch redundancy. The placeholder
    below is a farthest-first traversal in penultimate-embedding space,
    seeded by the labeled set, so every pick must be far from BOTH the
    labeled data and the picks made earlier in the same batch. It is
    deliberately uncertainty-blind — informativeness enters nowhere —
    which is exactly the headroom: combine the repulsion mechanic with a
    model-driven score (gradient magnitudes, margins, disagreement)
    without letting duplicates back into the batch. One formulation, no
    per-dataset constants.

    Interface (fixed): query(n) -> np.ndarray of n indices into self.X
    drawn from the currently unlabeled pool.
    """

    # Chunk size for pairwise-distance computation (memory bound, not tuning).
    dist_chunk = 4096

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def _min_dist_to(self, U, refs):
        """Min squared euclidean distance from each row of U to any row of
        refs, computed in chunks so the letter-sized pool fits in memory."""
        out = np.full(U.shape[0], np.inf)
        r2 = (refs * refs).sum(1)
        for s in range(0, U.shape[0], self.dist_chunk):
            blk = U[s:s + self.dist_chunk]
            d2 = (blk * blk).sum(1)[:, None] + r2[None, :] - 2.0 * blk @ refs.T
            out[s:s + self.dist_chunk] = d2.min(1)
        return out

    def query(self, n):
        """Greedy k-center batch: each successive pick maximizes its
        distance to the labeled set plus all previous picks."""
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        emb = self.get_embedding(self.X, np.asarray(self.Y)).numpy()
        U = emb[idxs_unlabeled]
        L = emb[self.idxs_lb]

        mind = self._min_dist_to(U, L)
        chosen = []
        for _ in range(int(n)):
            k = int(np.argmax(mind))
            chosen.append(int(idxs_unlabeled[k]))
            gap = U - U[k]
            mind = np.minimum(mind, (gap * gap).sum(1))
            mind[k] = -np.inf
        return np.asarray(chosen)

# ================================================================
# END EDITABLE REGION
# ================================================================

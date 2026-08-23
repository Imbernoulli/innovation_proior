class CustomSampling(Strategy):
    """Rare-class discovery: route labels toward starved classes.

    Variant objective: keep the labeled class histogram from starving any
    class the pool actually contains. The placeholder scores each
    unlabeled candidate by the posterior mass it places on classes that
    are under-supplied in the labeled set (deficit weighting), with a
    small uncertainty tiebreak. Its known weaknesses are the intended
    work: posteriors of a starved class are unreliable (the model can be
    blind to the class it most needs), so embedding/cluster evidence
    should back them up, and the deficit weight is static rather than
    proportional to measured starvation — on a well-covered binary pool
    it should fade toward plain informativeness.

    Interface (fixed): query(n) -> np.ndarray of n indices into self.X
    drawn from the currently unlabeled pool.
    """

    # Strength of the uncertainty tiebreak relative to deficit routing.
    tiebreak = 0.1

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        """Deficit-weighted acquisition: candidates likely to belong to
        under-labeled classes outrank merely-uncertain ones."""
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(
            self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled]
        ).numpy()
        n_cls = probs.shape[1]

        labeled_y = np.asarray(self.Y)[self.idxs_lb]
        counts = np.bincount(labeled_y.astype(int), minlength=n_cls).astype(float)
        # deficit: large for classes with few collected labels
        deficit = 1.0 / np.sqrt(1.0 + counts)
        deficit = deficit / deficit.sum()

        starve_score = probs @ deficit          # expected rare-class mass
        uncert = 1.0 - probs.max(axis=1)        # mild informativeness term
        score = starve_score + self.tiebreak * uncert
        return idxs_unlabeled[np.argsort(-score)[:int(n)]]

# ================================================================
# END EDITABLE REGION
# ================================================================

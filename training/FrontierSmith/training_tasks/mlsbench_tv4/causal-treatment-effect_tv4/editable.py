class CATEEstimator(BaseCATEEstimator):
    """Bagged, shrinkage-tuned CATE estimator (stability scaffold).

    Variant objective: the harness averages metrics over 5 folds x 10
    replications, so refit-to-refit variance inflates PEHE directly.
    This scaffold makes stability a constructed property:

      * shrinkage strength (ridge alpha) is picked by an internal
        K-fold rule on the pooled outcome regression -- a data-driven
        knob, not a constant, and never a per-dataset branch;
      * the effect surface is an AVERAGE over ``n_bags`` bootstrap
        refits of a T-Learner, so single-resample accidents cancel;
      * ``refit_dispersion_`` records the mean std of predictions
        across bags -- the measurable stability diagnostic the variant
        calls for.

    The base learner is linear on purpose: swap in richer learners only
    if the dispersion diagnostic shows the averaging can absorb their
    variance. Collapse-to-constant is the failure mode on the other
    side; PEHE will expose it.

    Interface contract (FIXED harness):
        fit(X, T, Y) -> self
        predict(X) -> tau_hat of shape (n,)

    Args:
        n_bags: bootstrap refits averaged at predict time.
        alphas: candidate ridge strengths for the internal selection.
    """

    def __init__(self, n_bags=10, alphas=(0.3, 1.0, 3.0, 10.0)):
        self.n_bags = int(n_bags)
        self.alphas = tuple(float(a) for a in alphas)
        self._models = []
        self._mean_tau = 0.0
        self.refit_dispersion_ = None

    def _pick_alpha(self, X, T, Y):
        """Small internal CV on the pooled outcome model (fast proxy)."""
        XT = np.column_stack([X, T.reshape(-1, 1)])
        best, best_err = self.alphas[0], np.inf
        kf = KFold(n_splits=3, shuffle=True, random_state=1)
        for a in self.alphas:
            err = 0.0
            for tr, te in kf.split(XT):
                m = Ridge(alpha=a).fit(XT[tr], Y[tr])
                err += float(np.mean((m.predict(XT[te]) - Y[te]) ** 2))
            if err < best_err:
                best, best_err = a, err
        return best

    def fit(self, X, T, Y):
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = X.shape[0]

        alpha = self._pick_alpha(X, T, Y)
        rng = np.random.RandomState(7)
        self._models = []
        for _ in range(self.n_bags):
            idx = rng.randint(0, n, size=n)  # bootstrap resample
            Tb, Xb, Yb = T[idx], X[idx], Y[idx]
            if not (Tb == 1).any() or not (Tb == 0).any():
                continue
            m1 = Ridge(alpha=alpha).fit(Xb[Tb == 1], Yb[Tb == 1])
            m0 = Ridge(alpha=alpha).fit(Xb[Tb == 0], Yb[Tb == 0])
            self._models.append((m1, m0))
        if not self._models:
            self._mean_tau = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if not self._models:
            return np.full(n, self._mean_tau)
        preds = np.stack([m1.predict(X) - m0.predict(X)
                          for m1, m0 in self._models])
        # Stability diagnostic: dispersion of the bagged surfaces.
        self.refit_dispersion_ = float(np.mean(preds.std(axis=0)))
        return preds.mean(axis=0)

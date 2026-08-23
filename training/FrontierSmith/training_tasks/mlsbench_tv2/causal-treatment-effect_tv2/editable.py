class CATEEstimator(BaseCATEEstimator):
    """Deviation-first CATE estimator (heterogeneity-tail scaffold).

    Variant objective: PEHE's squared-error mass concentrates on the
    individuals whose true effect is far from the mean, so the centered
    effect field tau(x) - mean is modeled as the PRIMARY object and
    flat prediction is treated as failure. Structure:

      1. plug-in stage -- per-arm outcome models give a rough effect
         surface at the training rows;
      2. deviation stage -- a small tree is fit to the CENTERED plug-in
         effects with sample weights amplified where the deviation is
         large (``tail_gain``), concentrating capacity in the tails;
      3. prediction = center + deviation model.

    The plug-in pseudo-effect channel is deliberately weak (ridge
    T-Learner): replacing it with an orthogonal / debiased pseudo-
    outcome while KEEPING the tail-weighted deviation emphasis is the
    intended direction of travel. ATE error is a constraint here, not
    the prize -- the center term keeps it from drifting.

    Interface contract (FIXED harness):
        fit(X, T, Y) -> self
        predict(X) -> tau_hat of shape (n,)

    Args:
        tail_gain: how strongly large-deviation rows are up-weighted in
            the deviation fit (0 = uniform weights).
        tree_depth: capacity of the deviation model.
    """

    def __init__(self, tail_gain=2.0, tree_depth=4):
        self.tail_gain = float(tail_gain)
        self.tree_depth = int(tree_depth)
        self._scaler = None
        self._center = 0.0
        self._dev_model = None

    def fit(self, X, T, Y):
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        # Stage 1: rough per-arm surfaces -> plug-in effects at train rows.
        mu1 = Ridge(alpha=1.0).fit(Xs[T == 1], Y[T == 1]) if (T == 1).any() else None
        mu0 = Ridge(alpha=1.0).fit(Xs[T == 0], Y[T == 0]) if (T == 0).any() else None
        if mu1 is None or mu0 is None:
            self._center = 0.0
            self._dev_model = None
            return self
        tau_plug = mu1.predict(Xs) - mu0.predict(Xs)

        # Stage 2: model the CENTERED field, weighting the tails up.
        self._center = float(np.mean(tau_plug))
        dev = tau_plug - self._center
        mad = float(np.median(np.abs(dev))) + 1e-12
        w = 1.0 + self.tail_gain * np.abs(dev) / mad
        self._dev_model = DecisionTreeRegressor(
            max_depth=self.tree_depth, min_samples_leaf=25, random_state=0
        ).fit(Xs, dev, sample_weight=w)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._dev_model is None or self._scaler is None:
            return np.full(X.shape[0], self._center)
        Xs = self._scaler.transform(X)
        return self._center + self._dev_model.predict(Xs)

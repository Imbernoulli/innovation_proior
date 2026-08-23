class CATEEstimator(BaseCATEEstimator):
    """ATE-anchored, overlap-governed CATE estimator (variant scaffold).

    Variant objective: PEHE and ATE error must fall TOGETHER. The class
    is structured around two explicit components:

      1. ATE anchor -- a stabilized (clipped Hajek/IPW) estimate of the
         average effect; the mean prediction cannot drift far from it.
      2. Overlap governor -- predictions shrink continuously toward the
         anchor wherever the estimated propensity approaches 0 or 1,
         instead of extrapolating an outcome model into a region where
         one arm is essentially unobserved. This doubles as the tail
         control that the log-scored economic setting demands.

    The heterogeneity component below is a deliberately weak linear
    T-Learner: replace it (orthogonalization, boosting, forests,
    residualization, ...) while keeping the anchor + governor structure
    doing real work.

    Interface contract (FIXED harness):
        fit(X, T, Y) -> self
        predict(X) -> tau_hat of shape (n,)

    Args:
        overlap_shrinkage: exponent on the overlap weight; 0 disables
            the governor, larger values shrink low-overlap regions
            harder toward the anchor.
        propensity_clip: symmetric clip applied to propensity scores in
            the anchor's IPW weights (tail control for the anchor).
    """

    def __init__(self, overlap_shrinkage=1.0, propensity_clip=0.05):
        self.overlap_shrinkage = float(overlap_shrinkage)
        self.propensity_clip = float(propensity_clip)
        self._prop = None
        self._mu0 = None
        self._mu1 = None
        self._ate_anchor = 0.0

    def _overlap_weight(self, e):
        """Continuous overlap diagnostic in [0, 1]: 4 e (1 - e), raised
        to ``overlap_shrinkage``. Near 1 in well-overlapped regions,
        near 0 where one arm is unobserved."""
        w = np.clip(4.0 * e * (1.0 - e), 0.0, 1.0)
        return w ** self.overlap_shrinkage

    def fit(self, X, T, Y):
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        # Propensity model: the governor's input.
        self._prop = LogisticRegression(max_iter=1000)
        self._prop.fit(X, T)
        e = np.clip(self._prop.predict_proba(X)[:, 1],
                    self.propensity_clip, 1.0 - self.propensity_clip)

        # ATE anchor: clipped Hajek estimator (self-normalized IPW).
        w1 = T / e
        w0 = (1 - T) / (1.0 - e)
        self._ate_anchor = float(
            np.sum(w1 * Y) / max(np.sum(w1), 1e-12)
            - np.sum(w0 * Y) / max(np.sum(w0), 1e-12)
        )

        # Heterogeneity component: weak linear T-Learner placeholder.
        self._mu1 = Ridge(alpha=1.0).fit(X[T == 1], Y[T == 1]) if (T == 1).any() else None
        self._mu0 = Ridge(alpha=1.0).fit(X[T == 0], Y[T == 0]) if (T == 0).any() else None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if self._mu1 is None or self._mu0 is None or self._prop is None:
            return np.full(n, self._ate_anchor)
        tau_raw = self._mu1.predict(X) - self._mu0.predict(X)
        e = np.clip(self._prop.predict_proba(X)[:, 1],
                    self.propensity_clip, 1.0 - self.propensity_clip)
        w = self._overlap_weight(e)
        # Shrink toward the anchor exactly where overlap is poor.
        return self._ate_anchor + w * (tau_raw - self._ate_anchor)

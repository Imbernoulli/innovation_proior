class CATEEstimator(BaseCATEEstimator):
    """Scale-equivariant CATE estimator (dimensionless scaffold).

    Variant objective: multiply every outcome by 1000 and the predicted
    effects must come out multiplied by 1000, with nothing else about
    the fit changing. Enforced by construction:

      * covariates are standardized with training-fold statistics;
      * outcomes are ROBUSTLY centered/scaled (median and a
        median-absolute-deviation-based yardstick, since heavy-tailed
        economic outcomes make raw standard deviations unstable);
      * the only model is fit entirely in the standardized space, where
        its hyperparameters are dimensionless;
      * predictions are back-transformed exactly once, at the exit.

    The in-space model is a plain S-Learner on [X, T, T*X] -- weak by
    intent. Improve it freely (interactions, nonlinearity, debiasing)
    but ONLY in the standardized space: any constant carrying outcome
    units re-introduces the currency mismatch this variant forbids.
    No branch on n, p, or dataset fingerprint anywhere.

    Interface contract (FIXED harness):
        fit(X, T, Y) -> self
        predict(X) -> tau_hat of shape (n,)

    Args:
        alpha: ridge strength -- dimensionless because the target is
            standardized before fitting.
        robust: robust (median/MAD) vs moment (mean/std) Y scaling.
    """

    def __init__(self, alpha=1.0, robust=True):
        self.alpha = float(alpha)
        self.robust = bool(robust)
        self._xscaler = None
        self._y_center = 0.0
        self._y_scale = 1.0
        self._model = None

    def _y_stats(self, Y):
        if self.robust:
            c = float(np.median(Y))
            s = 1.4826 * float(np.median(np.abs(Y - c)))  # MAD -> sigma units
        else:
            c, s = float(np.mean(Y)), float(np.std(Y))
        return c, (s if s > 1e-12 else 1.0)

    @staticmethod
    def _design(Xs, t):
        """[X, T, T*X]: lets the treatment shift AND tilt the surface."""
        tcol = np.full((Xs.shape[0], 1), float(t)) if np.isscalar(t) \
            else np.asarray(t, dtype=float).reshape(-1, 1)
        return np.column_stack([Xs, tcol, tcol * Xs])

    def fit(self, X, T, Y):
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        self._xscaler = StandardScaler().fit(X)
        Xs = self._xscaler.transform(X)
        self._y_center, self._y_scale = self._y_stats(Y)
        Yz = (Y - self._y_center) / self._y_scale  # dimensionless target

        self._model = Ridge(alpha=self.alpha).fit(self._design(Xs, T), Yz)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._model is None or self._xscaler is None:
            return np.zeros(X.shape[0])
        Xs = self._xscaler.transform(X)
        tau_z = (self._model.predict(self._design(Xs, 1.0))
                 - self._model.predict(self._design(Xs, 0.0)))
        # Exit-point back-transform: the ONLY place outcome units return.
        return tau_z * self._y_scale

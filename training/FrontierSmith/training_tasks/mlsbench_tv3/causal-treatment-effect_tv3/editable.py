class CATEEstimator(BaseCATEEstimator):
    """Cross-fitted doubly robust CATE estimator (orthogonality scaffold).

    Variant objective: nuisance errors may reach the effect estimate
    only through PRODUCTS of errors. The scaffold is a minimal
    DR-Learner:

      * internal K-fold cross-fitting -- propensity and per-arm outcome
        models are always evaluated on rows they were not trained on;
      * AIPW pseudo-outcome per row (outcome-model contrast plus
        inverse-propensity residual correction, propensity clipped);
      * final stage regresses the pseudo-outcome on X to produce
        tau_hat.

    Every component is the weakest reasonable choice (logistic
    propensity, ridge outcomes, ridge final stage) so that the
    orthogonal SKELETON is what carries the value; strengthening the
    stages -- and stress-testing each nuisance as the variant demands --
    is the intended work. Note the skeleton itself is what makes a
    single nuisance failure survivable; do not trade it away for a
    plug-in shortcut.

    Interface contract (FIXED harness):
        fit(X, T, Y) -> self
        predict(X) -> tau_hat of shape (n,)

    Args:
        n_folds: internal cross-fitting folds for the nuisances.
        clip: symmetric propensity clip inside the AIPW correction.
    """

    def __init__(self, n_folds=2, clip=0.02):
        self.n_folds = int(n_folds)
        self.clip = float(clip)
        self._final = None
        self._fallback = 0.0

    def fit(self, X, T, Y):
        X = np.asarray(X, dtype=float)
        T = np.asarray(T).astype(int).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = X.shape[0]

        pseudo = np.zeros(n)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
        for tr, te in kf.split(X):
            # Nuisances trained off-fold: the cross-fitting half of the
            # orthogonality contract.
            has_both = (T[tr] == 1).any() and (T[tr] == 0).any()
            if not has_both:
                pseudo[te] = float(np.mean(Y[tr])) if len(tr) else 0.0
                continue
            prop = LogisticRegression(max_iter=1000).fit(X[tr], T[tr])
            m1 = Ridge(alpha=1.0).fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
            m0 = Ridge(alpha=1.0).fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])

            e = np.clip(prop.predict_proba(X[te])[:, 1],
                        self.clip, 1.0 - self.clip)
            mu1, mu0 = m1.predict(X[te]), m0.predict(X[te])
            t, y = T[te], Y[te]
            # AIPW pseudo-outcome: errors enter as products, the
            # double-robustness half of the contract.
            pseudo[te] = (mu1 - mu0
                          + t * (y - mu1) / e
                          - (1 - t) * (y - mu0) / (1.0 - e))

        self._fallback = float(np.mean(pseudo))
        self._final = Ridge(alpha=1.0).fit(X, pseudo)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._final is None:
            return np.full(X.shape[0], self._fallback)
        return self._final.predict(X)

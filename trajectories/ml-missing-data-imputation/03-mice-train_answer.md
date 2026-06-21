KNN did exactly what I expected on the wide matrices and exactly what I feared on the narrow one. Breast Cancer RMSE fell from the mean fill's $0.994$ to $0.586$ — a 41% cut, the largest single jump on this ladder, because thirty correlated columns is precisely where reading the row pays — and Wine dropped to $0.796$, with the near-saturated downstream scores barely moving. But California *regressed* on both metrics, RMSE $0.933$ against the mean fill's $0.928$ and R² $0.612$ against $0.646$, and the per-seed numbers $\{0.941, 0.937, 0.922\}$ are tight enough that this is structural, not a lucky seed. The diagnosis I wrote holds: with only eight columns the overlap set per pair is tiny, the $d/|O|$ scale-up extrapolates a full-dimensional distance from six or seven coordinates, and a local average over noisy neighbours of a genuinely continuous, weakly-redundant target is worse than the stable column mean. KNN is the right idea — use the row — built with the wrong machinery for narrow continuous tables: it *copies* nearby rows, so when no row is reliably close it has nothing.

The fix is a change of mechanism, not a tweak. I propose **MICE — multiple imputation by chained equations**, in the lineage of van Buuren & Groothuis-Oudshoorn (2011), but in its conditional-mean (point) form. Instead of copying nearby rows, model each incomplete column as a function of all the others and predict its holes: to fill column $j$, regress $Y_j$ on the other columns over the rows where $Y_j$ was observed, then predict the missing rows. This borrows strength from every feature at once rather than leaning on whether a handful of rows happen to be close — exactly what California needed, where the eight features carry real joint signal even though no two rows are near-twins. The licence is MCAR: the missingness mechanism is ignorable, so the conditional distribution of a missing entry given the observed is the right thing to predict from, and a regression of $Y_j$ on the rest estimates that conditional.

The obstacle is that "regress $Y_j$ on the other columns" hits a tangle the moment I write it: the other columns are themselves full of holes, and the dependence is circular — income would help fill house age, but house age would equally help fill income, and no column is complete enough to treat as a fixed predictor. KNN sidestepped this pairwise; a single global regression cannot, because it needs a full design matrix. The way around is to break the deadlock with a rough start and then refine. Fill *every* column's holes first with its observed mean — the best no-covariate guess, which I already have — just to stand on a complete table. Now every column is crudely complete, so I can run the single-column regression for each: take column $j$, discard its current crude fill, regress it on the *current* values of all the other columns over the rows where $j$ was genuinely observed, and re-impute $j$'s holes from that fit. Move to the next column, regress it on the now-updated others including the freshly re-imputed $j$, re-impute. One pass through all the columns is a sweep; then sweep again, and again. The chained structure is what dissolves the circularity: nobody is ever a fixed complete predictor, everybody is conditionally re-estimated given everybody else's current state, repeatedly.

It is worth naming what this iteration actually is, because misnaming it would mis-tune it. Re-imputing each block $Y_j$ from its conditional given all the others, cycling through the blocks, is a Gibbs sweep over the per-column conditionals. The reassuring consequence is about mixing: when I re-impute $Y_j$ at one sweep I condition on the current values of the other columns but throw away $Y_j$'s own previous fill entirely, so last sweep's $Y_j$ influences this sweep's only indirectly, through how it changed the *other* columns that feed back as predictors. That is a weak coupling, so the chain decorrelates fast — the dependence on the crude mean initialization washes out in a handful of sweeps rather than hundreds, which is what makes the scheme cheap enough to run inside the loop.

Now the per-column engine, and here the benchmark's shape decides it. The task does not ask for valid statistical inference with pooled standard errors across multiple imputations; it asks for *one* completed matrix scored on squared-error RMSE and a downstream model. For squared error the single fill that minimizes expected loss is the *conditional mean* of the missing value given the observed — not a random draw from the conditional distribution. A proper multiple-imputation draw deliberately adds residual scatter and parameter jitter around that mean, which is exactly what makes the between-imputation variance correct for inference, but that same scatter moves the fill off the variance-minimizing mean and *increases* squared error against a fixed truth. So I take the point version: at each column, regress and impute the predicted conditional mean, with no residual noise and no posterior draw (`sample_posterior=False`). The iteration over columns stays — that is what handles the multivariate circularity and borrows cross-feature strength — but each step is a deterministic conditional-mean regression. That decision also fixes the regressor, and I want the self-tuning kind so I never hand-pick a ridge per column across three heterogeneous tables. Plain least squares would be unstable on the wide collinear Breast Cancer matrix — thirty near-duplicate features make $X'X$ badly conditioned — and a fixed ridge would need a penalty I would have to guess differently per dataset. The clean answer is `BayesianRidge`: a Bayesian linear regression that puts a Gaussian prior on the coefficients and a noise precision and estimates both precisions from the data by evidence maximization, so the effective ridge strength is the ratio of the two and is chosen automatically — a collinear or small-$n$ column shrinks more, a well-determined one less, with no manual tuning. Its posterior-mean prediction is precisely the conditional mean I want for squared-error reconstruction.

The sweep order follows the same reliability argument that drove KNN's neighbour selection. The per-column conditionals, chosen independently, need not be mutually consistent — there may be no single joint distribution with all of them as its conditionals — and when they are incompatible the visit order matters. The sensible order is ascending in missingness: impute the least-incomplete columns first, because they have the most real observed rows to fit from and give the most reliable predictions, so by the time I reach the heavily-missing columns their predictors are already as good as they will get. Under uniform 20% MCAR the columns are roughly equally missing, so the ordering is mild here, but it costs nothing and is the right default. Assembled, the loop is: initialize every column's holes with its observed mean; visit columns ascending in missingness; for a capped number of rounds, sweep, fitting `BayesianRidge` on the genuinely-observed rows of each incomplete column against the others' current fill, predicting and writing back so the next column sees the update immediately; after each sweep, if the imputed entries changed by less than a small tolerance relative to the data scale, stop early, since the fixpoint is effectively reached. I set the cap at thirty sweeps with a scaled-change tolerance of `1e-3` — generous, because the fast-mixing argument says the early stop usually fires first — and the originally observed entries are restored untouched.

This is exactly scikit-learn's `IterativeImputer`, which gives the round-robin shell, the mean initialization, the ascending order, the per-feature fit/predict, the `max_iter` cap, and the tolerance early-stop, so the faithful edit is `IterativeImputer(estimator=BayesianRidge(), max_iter=30, imputation_order="ascending", initial_strategy="mean", tol=1e-3)`. California is the test of the whole move — a linear model using all eight features jointly should beat the mean fill, RMSE below $0.928$ and R² above $0.646$, undoing KNN's regression — and Breast Cancer should hold or improve below KNN's $0.586$, since a self-regularizing regression on thirty correlated features is well-suited and the conditional mean of a strongly-predictable column is sharper than a neighbour average. The open risk is Wine: with only 178 rows each per-column regression fits on roughly 140 observed rows against twelve predictors, a thin fit, and a *linear* conditional mean will miss any genuinely non-linear feature relationship that KNN's local average could have followed by bending to the data — so Wine's RMSE may lag KNN's $0.796$. If it does, the next rung's diagnosis is written: the chained-equations skeleton is right, but a *linear* per-column model is the wrong predictor when the relationships are non-linear, and I will swap it for one that captures non-linearities and interactions on its own.

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    """MICE: Multiple Imputation by Chained Equations.

    Uses sklearn.impute.IterativeImputer with BayesianRidge estimator.
    Reference: van Buuren & Groothuis-Oudshoorn (2011).
    """

    def __init__(self, random_state=42, max_iter=30):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
        )
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
        )
        return self._imputer.fit_transform(X)


def compute_feature_correlations(X):
    n_features = X.shape[1]
    corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
                corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    return corr
```

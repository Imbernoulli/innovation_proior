Almost every analysis I want to run on a real biological table refuses to start while the table has holes in it: regression drops incomplete rows, PCA and clustering need a full matrix, and a distance is undefined the moment one coordinate is missing. So before anything else I must turn an $n \times p$ matrix $X$ with $\mathrm{NaN}$ scattered through it into a finite-valued completion $X_{\mathrm{imp}}$ — and not just any completion, but one whose filled-in numbers are close to the truth I cannot see, so that whatever runs downstream behaves as if the data had been observed. The cheapest fill, replacing each hole with its column mean, uses nothing about the rest of the row: if two variables are tightly related, the mean-filled value sits at the column average regardless of what the other variables say, exactly where the information was. Mean imputation therefore flattens the structure I am trying to preserve — it deflates the imputed column's variance and washes out its correlations — so it is the baseline to beat, not the answer. The real signal is that a missing entry is predictable from the other variables, and the job is to exploit that. The trouble is that real biological and medical data have, all at once, mixed variable types (continuous measurements sitting next to categorical labels, with cross-type relations that carry information), nonlinear dependencies and interactions, variables on wildly different scales, sometimes $p \gg n$, and no prior knowledge to guide a modeling choice or a hyperparameter.

Each prior method stalls on this same wall from a different side. KNNimpute (Troyanskaya et al. 2001) imputes a hole by a distance-weighted average over the $k$ nearest rows on the observed coordinates; it is local and assumption-light, but it is defined for continuous data only, its Euclidean distance forces me to standardize variables to a common scale and de-standardize afterward, its neighbor count $k$ is a knob with a large effect on accuracy that I would have to cross-validate, and a weighted neighbor average is locally smooth — nearly linear — so it blurs sharp nonlinearities and interactions. The chained-equations idea (MICE; Van Buuren & Oudshoorn 1999) is the cleverest framing available: instead of writing down one joint distribution over all $p$ variables — brutal to specify for mixed types and often nonexistent in clean parametric form — it specifies for each incomplete variable a conditional model of that variable given all the others, then cycles round-robin, re-imputing one variable from the current fill of the rest and moving on. That skeleton is exactly the right shape and I want to keep it. But MICE makes me supply a parametric conditional family for every single variable (linear here, logistic there, polytomous somewhere else), the choice matters and I usually have no basis for it, it assumes those conditionals are draws from an implied joint distribution that exists, and on ill-behaved variables the implementation falls over unless I prune by hand. The penalized route MissPALasso (Städler & Bühlmann 2010) is an EM over lasso regressions of missing-on-observed — continuous-only, linear-Gaussian, with a penalty $\lambda$ to cross-validate, and infeasible at very high dimension. The matrix-completion route, the iterative soft-thresholded SVD (Mazumder, Hastie & Tibshirani 2010), is structurally elegant but imposes a single global low-rank linear model on the whole matrix and is continuous-only. Every one carries a restriction (one variable type), or a knob I must guess ($k$, $\lambda$), or a required preprocessing (standardization), or a structural assumption (linear, Gaussian, low-rank) that throws away the nonlinear, interaction-laden, mixed-type structure that the data is made of.

So I keep the chained-equations round-robin and change only what goes into each per-variable slot. The predictor I need there must, with no help from me, handle continuous and categorical inputs at once, ignore scale, capture nonlinearities and interactions on its own, assume no parametric family, and be robust enough that I need not nurse it. That spec sheet is the spec sheet of a random forest, and I propose to plug one into each slot — this is MissForest. A random forest (Breiman 2001) grows many unpruned CART trees, each on a bootstrap resample of the rows, searching at every node for the best split only among a random subset of $m_{\mathrm{try}}$ variables, and averages the trees for regression (votes for classification). CART splits ask "is variable $k$ below a threshold" for a numeric variable or "is $k$ in this set of levels" for a categorical one, so a single tree natively mixes types and is invariant to any monotone rescaling — no standardization, ever; the recursive partition represents interactions and nonlinear surfaces for free; and there is no distributional family to choose. The forest checks every box on the spec sheet at once.

Making "regress variable $X_s$ on the others" concrete runs into an immediate snag: the other columns at the rows where $X_s$ is observed are themselves full of holes, so I cannot simply fit a forest on them. The way out is to keep a fully completed working matrix at all times. I start by filling every hole cheaply — column means for numeric variables, modal class for categorical ones — so that from the very first sweep every predictor block is finite and a forest can be fit; that initial guess only needs to be complete, not clever, because the iteration is what refines it. Then within a sweep, when I impute $X_s$, I train the forest with response $y_{\mathrm{obs}}$, the genuinely observed values of $X_s$ (I never train on my own guesses for $X_s$), and inputs the other columns of the current working matrix restricted to the observed-$s$ rows; I predict the missing $X_s$ from the other columns at the missing-$s$ rows; and I overwrite those holes in the working matrix. I then move to the next variable using the matrix I just updated — Gauss-Seidel, not Jacobi — so an improved earlier variable immediately helps the regression of a later one within the same pass, which is why a couple of sweeps already mix structure across all variables. The order within a sweep is not arbitrary: a forest is only as good as its training set, so I sort the variables by ascending missingness and impute the most-complete variable first. It has the most observed rows to learn from and the fewest holes to corrupt later regressions, and once improved its column becomes a higher-quality predictor for the harder, more-missing variables later in the same sweep. The opposite order would have me predicting the most-missing variable from everyone else's still-mostly-mean-filled columns, the worst possible inputs.

The stopping rule reads the imputation's own dynamics rather than introducing yet another knob. After each sweep I measure how much the working matrix changed. For the continuous variables I use the relative, dimensionless squared change

$$\Delta_N = \frac{\sum_{j\in N}(X_{\mathrm{new}} - X_{\mathrm{old}})^2}{\sum_{j\in N}(X_{\mathrm{new}})^2},$$

and for the categorical variables the fraction of categorical holes whose predicted label flipped,

$$\Delta_F = \frac{\#\{\text{categorical entries changed}\}}{\#\{\text{missing categorical entries}\}}.$$

A random forest is not a noise-free map: bootstrap resampling and the random feature subset make the prediction stochastic, so the imputation never converges to a single fixed matrix — it descends toward a good region and then jitters. Early on $\Delta$ is large and falls as the fill settles; once inside the jitter the sweep-to-sweep change stops decreasing and bounces back up. That minimum is the signal. So the rule is not "$\Delta$ below a threshold I would have to pick" but "keep sweeping while $\Delta$ is still decreasing, and stop the first time $\Delta$ increases," returning not the current sweep (which already overshot into the jitter) but the previous sweep's matrix, the one at the minimum. With both types present I track $\Delta_N$ and $\Delta_F$ separately and stop only once both have turned up. A maximum number of sweeps is a safety cap in case the change never rises; on continuous data the rule typically fires after about five sweeps, so the cap rarely ends it.

The forest's own knobs are not tuned either. The first is $m_{\mathrm{try}}$. Why not search over all $p$ variables at every node to make each tree as strong as possible? Because the forest's error is not the average tree's error — averaging is what helps — and Breiman's bounds make the lever precise: for classification $PE^* \le \bar\rho\,(1-s^2)/s^2$ and for regression $PE^*(\text{forest}) \le \bar\rho \cdot PE^*(\text{tree})$, where $s$ is individual-tree strength and $\bar\rho$ is the average correlation between the trees' errors. Both say: keep trees strong, but reduce $\bar\rho$. If every tree saw all variables at every node they would find nearly the same splits, $\bar\rho$ near 1, and averaging would buy almost nothing; restricting each node to a random $m_{\mathrm{try}}$-subset decorrelates the trees and drives $\bar\rho$ down, at a limited cost to individual strength. Breiman found the forest error remarkably flat in $m_{\mathrm{try}}$ with one or two features often near optimal, so I take the standard default $m_{\mathrm{try} } = \lfloor\sqrt{p}\rfloor$ — well below $p$, comfortably above the degenerate $m_{\mathrm{try}}=1$ (which forces the split variable and is no longer a forest), and going toward $p$ only raises $\bar\rho$ and runtime for little gain. The second knob is the number of trees. By the strong law of large numbers the forest error converges as trees are added — more trees never overfit, they only cost runtime, roughly linear in their count — so this is a precision-versus-cost knee, and about 100 trees sits where accuracy has flattened. The trees are grown deep and unpruned (low bias, with the variance absorbed by averaging over the hundred), with a small minimum leaf size of about five in regression so leaf means are not single noisy points, and bootstrap row sampling both decorrelates the trees and, for free, yields an error estimate. That bonus answers a question I would otherwise have no clean answer to — how good is my imputation when I have no truth for the filled entries? Each tree's size-$n$ bootstrap omits any fixed row with probability $(1-1/n)^n \to 1/e \approx 36.8\%$, so every row is out-of-bag for about $36.8\%$ of the trees; predicting it from only those trees gives an honest held-out prediction, and aggregating the out-of-bag squared errors over the observed-response rows gives an essentially unbiased internal error estimate per variable. Since a forest is fit per variable anyway, averaging the per-variable OOB MSE over the continuous variables (normalized) and the OOB misclassification over the categorical ones estimates the imputation error itself with no held-out set and no cross-validation.

The whole thing coheres as block coordinate descent on the imputed matrix: each variable's holes are a block, re-imputing block $s$ from the current values of all other blocks is a coordinate update toward mutual consistency, the Gauss-Seidel ordering propagates information faster than freezing the old matrix, the complete mean fill makes every update well-defined from sweep one, the ascending-missingness order puts the most reliable updates first, and the change-turns-up rule reads off the point where further sweeps would only churn forest noise. The quiet headline is that none of $m_{\mathrm{try}} = \lfloor\sqrt{p}\rfloor$, the hundred trees, or the difference-based stop is anything the analyst must tune — the entire contrast with the $k$ of KNNimpute and the $\lambda$ of the penalized methods. Concretely, for continuous tables the per-variable predictor is a `RandomForestRegressor` and the convergence test is the continuous $\Delta$; the categorical case is the direct analogue with modal-class initialization, a `RandomForestClassifier` per factor response, and the $\Delta_F$ test.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor


class MissForest(BaseEstimator, TransformerMixin):
    """Iterative random-forest imputation (MissForest).

    Chained-equations round-robin with a random forest as the per-variable
    predictor: complete mean fill, then sweep variables in increasing order of
    missingness, fitting a forest on each variable's observed part against the
    current values of the others and overwriting its holes; stop when the
    sweep-to-sweep change first increases (return the previous iterate).
    """

    def __init__(self, random_state=42, max_iter=10, n_estimators=100):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = n_estimators

    def fit(self, X, y=None):
        self._impute(X)           # learn from observed entries only; no test labels
        return self

    def transform(self, X):
        return self._impute(X)

    def fit_transform(self, X, y=None):
        return self._impute(X)

    def _impute(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        missing = np.isnan(X)                         # original missingness pattern

        # initial complete fill: column means (best no-covariate constant guess)
        col_mean = np.nanmean(X, axis=0)
        X_imp = X.copy()
        for j in range(n_features):
            X_imp[missing[:, j], j] = col_mean[j]

        # variables to impute, sorted by ascending missingness
        miss_count = missing.sum(axis=0)
        order = [j for j in np.argsort(miss_count) if miss_count[j] > 0]
        if not order:
            return X_imp

        mtry = max(1, int(np.floor(np.sqrt(n_features))))  # canonical m_try = floor(sqrt(p))
        X_best = X_imp.copy()
        delta_old = np.inf
        for _ in range(self.max_iter):
            X_prev = X_imp.copy()

            for j in order:                           # Gauss-Seidel sweep
                obs = ~missing[:, j]
                mis = missing[:, j]
                others = [k for k in range(n_features) if k != j]
                if not others:
                    X_imp[mis, j] = np.nanmean(X[obs, j])
                    continue

                X_train = X_imp[obs][:, others]       # current fill of other columns
                y_train = X[obs, j]                   # genuinely observed targets
                X_pred = X_imp[mis][:, others]

                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,   # ~100: error flattened, runtime ~linear
                    max_features=min(mtry, len(others)),
                    min_samples_leaf=5,                # R randomForest regression nodesize default
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                X_imp[mis, j] = rf.predict(X_pred)

            denom = np.sum(X_imp ** 2)
            delta_new = np.sum((X_imp - X_prev) ** 2) / denom if denom > 0 else 0.0
            if delta_new > delta_old:                 # change turned up -> stop
                return X_best                         # the iterate at the minimum
            X_best = X_imp.copy()
            delta_old = delta_new

        return X_best                                 # cap reached: latest iterate
```

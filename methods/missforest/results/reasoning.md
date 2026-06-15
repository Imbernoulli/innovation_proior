Let me start from what actually breaks when I hand someone a real biological data table. It's an `n × p` matrix with holes scattered through it — a microarray with dropped genes, a clinical registry where some patients skipped a question, a mass-spec run with a few settings unrecorded — and almost every analysis I'd want to run next refuses to start. Regression drops incomplete rows; PCA and clustering need a full matrix; a distance is undefined the moment one coordinate is missing. So before anything, I have to produce a finite-valued completion `X_imp`. And not just any completion — I want one whose filled-in numbers are close to the truth I can't see, so that whatever I run downstream behaves as if the data had been observed.

The cheapest thing is to fill each hole with that column's mean. But the mean uses *nothing about the rest of the row*. If two variables are tightly related, the mean-filled value sits at the column average regardless of what the other variables say, which is exactly where the information was. So mean imputation flattens out the very structure I'm trying to preserve — it deflates the imputed column's variance and washes out its correlations. It's the baseline, the thing I have to beat, not the answer. The real signal is that a missing entry is *predictable from the other variables*, and the job is to exploit that.

So who already predicts a missing entry from the others, and where does each one stop? The neighborhood approach — KNNimpute, Troyanskaya and colleagues — for a hole in some variable, finds the `k` rows most similar on the observed coordinates and imputes a distance-weighted average of their values. I like that it's local and assumption-light. But it bites me three ways here. First, it's built for continuous data; my tables have categorical columns sitting right next to continuous ones, and a Euclidean distance over a mix of "expression level 4.7" and "disease state = yes" isn't even well defined. Second, that distance forces me to standardize — variables on wildly different scales would otherwise let the large-scale ones dominate the neighbor search, so I have to rescale everything to unit variance, impute, then de-scale, and babysit that pipeline. Third, and worst, `k` is a knob with a big effect on accuracy and no way to know it in advance, so in practice I'd have to cross-validate it — laborious, and the whole point was supposed to be "I don't know much about this data." And a weighted average of neighbors is locally smooth, basically linear in the neighborhood; sharp nonlinearities and interactions between variables get blurred.

What about a model-based route that handles mixed types? The chained-equations idea — MICE, van Buuren and Oudshoorn — is the cleverest framing I have. The trap I'd otherwise fall into is trying to write down one joint distribution over all `p` variables and impute from it; for mixed continuous-and-categorical data that joint model is brutal to specify and often doesn't exist in any clean parametric form. Chained equations sidesteps the joint entirely: specify, for each incomplete variable, a *conditional* model of that variable given all the others — linear regression for a continuous column, logistic for a binary one, polytomous logistic for a multi-level one. Then cycle: pick a variable, regress it on the current best fill of the rest, replace its holes with draws from that conditional, move to the next variable using the values I just updated, and go around again. Because each variable's update uses the freshly imputed earlier variables, the whole sweep behaves like a Gibbs sampler walking through the space of imputed values, and — this is the empirically pleasant part — it settles down in only a handful of cycles, five to twenty, partly because the random noise in the draws keeps successive states from being too correlated and helps it mix.

This chained-equations skeleton is genuinely the right shape — I want to keep it. But look at what MICE makes *me* supply: a parametric conditional model for every single variable. Linear here, logistic there, polytomous somewhere else. The choice matters — pick the wrong family and the imputation suffers — and I usually have no prior knowledge to make that choice with. On top of that it assumes those conditionals are all draws from some implied joint distribution that exists, and on ill-behaved variables (a binary column with three cases in one category, two nearly dependent columns) the implementation just falls over unless I prune those variables by hand. So MICE solves mixed types but trades me a pile of modeling decisions and assumptions I can't responsibly make. The penalized-regression route, MissPALasso, has the same flavor of cost — it's an EM over lasso regressions of missing-on-observed, continuous-only, linear-Gaussian, and it carries a penalty `λ` I'd have to cross-validate, plus it chokes at very high dimension. And the matrix-completion route, the iterative soft-thresholded SVD of Mazumder, Hastie and Tibshirani, is structurally beautiful — fill the holes, refit a low-rank approximation, refill, iterate — but it imposes a single *global low-rank linear* model on the entire matrix and is continuous-only, so per-variable nonlinear dependence and categorical columns are out of reach.

Stepping back, every one of these stalls on the same wall from a different side: each carries a restriction (one variable type), or a knob I must guess (`k`, `λ`), or a required preprocessing (standardization), or a structural assumption (linear, Gaussian, low-rank) that throws away the nonlinear, interaction-laden, mixed-type structure that real biological data is *made of*. So the chained-equations skeleton stays, but the thing I plug into each per-variable slot has to be different. What I want in that slot is a predictor that, with no help from me, handles continuous and categorical inputs at once, doesn't care about scale, captures nonlinearities and interactions on its own, needs no parametric family specified, and ideally is robust enough that I don't have to nurse it. Stated that way, the requirements list reads like a spec sheet — and there's a method whose spec sheet it is.

A random forest. Breiman's. Let me actually check it against the list rather than just asserting it. It grows many unpruned CART trees, each on a bootstrap resample of the rows, and at each node it searches for the best split only among a random subset of `m_try` of the variables; for prediction it averages the trees (regression) or lets them vote (classification). CART splits ask "is variable `k` below a threshold" for a numeric variable or "is variable `k` in this set of levels" for a categorical one — so a single tree natively mixes continuous and categorical *predictors*, and a split is invariant to any monotone rescaling of a variable, which means no standardization, ever. The tree structure is a recursive partition, so it represents nonlinear response surfaces and interactions for free — an interaction is just a split on one variable nested inside a split on another. There's no parametric family to choose. So a forest checks every box on the spec sheet at once, and that can't be a coincidence — it's the same nonparametric, mixed-type, assumption-light character I wrote down as the goal. So the move is: take the chained-equations round-robin, and make the per-variable conditional predictor a random forest.

Now I have to make this concrete, because there's an immediate snag with "regress variable `s` on the others." Let me lay out the four pieces for a single variable `X_s` with missing entries at rows `i_mis`. There's the observed part of `X_s` itself, `y_obs` — those are my training targets. There's the missing part `y_mis` — what I want to predict. There are the *other* variables at the rows where `X_s` is observed, `x_obs` — my training inputs. And the other variables at the rows where `X_s` is missing, `x_mis` — the inputs I'll predict from. But here's the snag: `x_obs` and `x_mis` are themselves full of holes, because the rows where `X_s` happens to be observed are not magically complete in the *other* columns. So I can't just "fit a forest on `x_obs`" — the inputs have `NaN`s in them. Wall.

The way out is the same trick the chained-equations skeleton was already using and I almost glossed over: keep a *fully completed working matrix* at all times. Start by filling every hole with something cheap — column means for numeric columns, the most frequent level for categorical ones — so that from the very first sweep, every predictor block is finite and a forest can be fit. Those initial fills are wrong, of course, but they're just a starting point; the iteration is what refines them. So the initial guess doesn't need to be clever, it only needs to be *complete*; the mean is fine because it's the best no-covariate constant and it's free. Now within a sweep, when I impute variable `s`, I train the forest with response `y_obs` (the genuinely observed values of `X_s` — I never train on my own guesses for `X_s`) and inputs the other columns of the *current working matrix* restricted to the observed-`s` rows; then I predict `y_mis` from the other columns at the missing-`s` rows; then I overwrite those holes in the working matrix with the predictions. Move to the next variable using the matrix I just updated — Gauss-Seidel, not Jacobi, so that the improved imputation of an earlier variable immediately helps the regression of a later one within the same pass. That within-sweep information flow is exactly why a couple of sweeps already mix structure across all the variables, the same reason MICE converges fast.

There's a choice I glossed: in what *order* do I take the variables within a sweep? It shouldn't be arbitrary. The forest for a variable is only as good as its training set, and the cleanest training set belongs to the variable with the *fewest* missing values — it has the most observed rows to learn from, and the fewest holes in its own column to corrupt later regressions. So sort the variables by amount of missingness, ascending, and impute the most-complete variable first. That does two good things: the easiest, best-trained forests run first, and their now-improved columns become higher-quality predictors for the harder, more-missing variables later in the same sweep. Doing it the other way — hardest first — would have me predicting the most-missing variable using everyone else's still-mostly-mean-filled columns, the worst possible inputs. Ascending it is.

So one full sweep is: for each variable in increasing-missingness order, fit a forest on its observed part against the current other columns, predict its holes, write them back. Then sweep again. When do I stop? I need a convergence test, and I want it to be about the imputation itself, not some external validation. The natural quantity is how much the working matrix *changed* from the previous sweep to this one. For the continuous variables, measure the total squared change normalized by the total squared magnitude,

  Δ_N = Σ_{j∈N} (X_new − X_old)² / Σ_{j∈N} (X_new)²,

so it's a relative, dimensionless number; and for the categorical variables, the fraction of categorical holes whose predicted label flipped between sweeps,

  Δ_F = (number of categorical entries that changed) / (number of missing categorical entries).

Both are near zero when the imputation has stopped moving. Now, the obvious stopping rule is "stop when Δ drops below some tolerance." But let me think about what Δ actually does over sweeps, because a random forest is not a noise-free map. Early on, the imputation is far from any fixed point and each sweep changes it a lot, so Δ is large. As the sweeps refine the fill, Δ falls — the matrix is settling. But the forest predictions carry irreducible randomness (bootstrap resampling, the random feature subset at each node), so the imputation never converges to a single fixed matrix; it descends toward a good region and then *jitters*. Once it's in that region, the sweep-to-sweep change stops decreasing and starts bouncing back up — Δ hits a minimum and then rises. That minimum is the signal. The best imputation is the one at the bottom, the sweep where the matrix moved the least; once Δ turns back up, the next sweeps are just shuffling noise.

So the rule isn't "Δ below a threshold I'd have to pick" — which would be yet another knob, and against the whole no-tuning ethos — it's "keep sweeping while Δ is still *decreasing*, and stop the first time Δ *increases*." And when it increases, I don't return the current sweep (which already overshot into the jitter); I return the *previous* sweep's matrix, the one that produced the minimum. For mixed data with both types present, I track Δ_N and Δ_F separately and only stop once *both* have turned up — as long as either type is still improving, keep going. I keep a safety cap, a maximum number of sweeps, in case the difference somehow never turns up; if I hit the cap, I just return the latest. In practice on continuous data this fires after only about five sweeps, so the cap is rarely the thing that ends it. And notice what I did *not* have to introduce: no tolerance to tune, no held-out validation set — the stopping rule reads the imputation's own dynamics.

Now let me settle the forest's own knobs, because if I have to hand-tune those I've reintroduced exactly the kind of cost I was fleeing. The first is `m_try`, the number of variables considered at each split. Why not just search over all `p` variables at every node, which would make each individual tree as strong as possible? Breiman's analysis is the reason, and it's worth doing rather than asserting. The forest's error isn't the average tree's error — averaging is what helps — and the bounds make the mechanism precise. In classification, if `s` is strength and `ρ̄` is the average correlation, `PE* ≤ ρ̄(1 − s²)/s²`; in regression, `PE*(forest) ≤ ρ̄ · PE*(tree)`, where `ρ̄` is the weighted correlation between residuals from two independently grown trees and `PE*(tree)` is the average single-tree error. In both cases the same lever appears: keep individual trees strong, but make their errors less correlated. If I let every tree see all variables at every node, they'd all find nearly the same splits and be highly correlated — `ρ̄` near 1 — and averaging would buy me almost nothing. Restricting each node to a random `m_try`-subset *decorrelates* the trees, pushing `ρ̄` down; the cost is that each tree is individually a bit weaker, nudging the strength down or the single-tree error up. The sweet spot is a small `m_try` — Breiman found the forest error remarkably flat in `m_try`, with one or two features often near optimal. The standard default that captures "small, decorrelating, but not degenerate" is `m_try = floor(sqrt(p))`. I should check the degenerate end: `m_try = 1` means there's no *choice* of variable at a node — the split variable is forced — so it isn't really a random forest anymore, just randomly-built trees, and the imputation error jumps, especially with few trees. So `sqrt(p)` it is: well below `p`, comfortably above 1. Going larger toward `p` only raises `ρ̄` and the runtime while barely moving accuracy — a bad trade.

The second knob is the number of trees. Here the law of large numbers is on my side: Breiman shows the forest's error *converges* as the number of trees grows — adding trees never overfits, it only tightens the estimate toward its limit. So more trees is monotonically (weakly) better in accuracy; the only cost is runtime, which is roughly linear in the number of trees. That makes the choice a precision-versus-cost knee, not a bias-variance tradeoff: pick enough trees that accuracy has flattened out but I'm not paying for trees that no longer help. A hundred trees per forest sits right at that knee — accuracy has essentially stagnated by then while runtime keeps climbing past it. So a hundred trees.

The trees themselves I grow deep and unpruned — that's Breiman's prescription and it's right here too: a deep tree has low bias (it can carve out fine structure), and the variance that deep trees would suffer in isolation is exactly what the averaging over a hundred trees absorbs. In regression the terminal nodes get a small minimum size (around five observations) so the leaf means aren't computed from a single noisy point, and bootstrap resampling (sampling rows with replacement) is what both decorrelates the trees and, as a bonus I'll come back to, gives me an error estimate for free.

That bonus is worth pausing on, because it solves a problem I'd otherwise have no clean answer to: how good *is* my imputation, when by definition I don't have the truth for the entries I just filled? With most methods I'd have to set aside known entries as a test set or run a cross-validation — wasteful of scarce data and laborious. But the forest hands me an estimate without either. Each tree is trained on a bootstrap sample of the rows, which omits any fixed row with probability `(1 − 1/n)^n → 1/e ≈ 0.368`, so every row is out-of-bag for about 36.8% of the trees. Predict that row using only the trees that didn't see it, and I get an honest held-out prediction; aggregate the squared error of these out-of-bag predictions over the observed-response rows and I have the out-of-bag error of that per-variable forest — an internal, essentially unbiased estimate of its generalization error, as accurate as a same-size test set. Since I fit a forest per variable anyway, I get an OOB error per variable for free. Aggregate them by type — average the per-variable OOB mean-squared errors over the continuous variables and normalize, average the OOB misclassification rates over the categorical ones — and I have an estimate of the imputation error itself, with no test set and no cross-validation. It's a real estimate of how trustworthy each completed column is. (To compare imputers when the truth *is* known, by artificially deleting entries, the accepted continuous yardstick is the normalized RMSE, `NRMSE = sqrt( mean((X_true − X_imp)²) / var(X_true) )` over the deleted entries — RMSE rescaled by the spread so 0 is perfect and ≈1 is no better than the mean — and for categorical entries the proportion falsely classified.)

Let me make sure the whole thing actually coheres as a fixed-point iteration before I write it down. I'm doing block coordinate descent on the imputed matrix: each variable's holes are a block, and re-imputing block `s` by regressing on the current values of all other blocks is a coordinate update toward mutual consistency — each variable made as predictable as possible from the rest. The Gauss-Seidel ordering (use the latest values within a sweep) propagates information faster than freezing the old matrix would. Starting from the complete mean fill guarantees every update is well-defined from sweep one. The ascending-missingness order makes the early, most-reliable updates happen first so the later ones have better inputs. And the stop-when-the-change-turns-up rule reads off the point where further sweeps would only be churning forest noise. Every piece earns its place; nothing is a free parameter the user has to set. That last property is the quiet headline — `m_try = floor(sqrt(p))`, a hundred trees, the difference-based stop — none of them is something I ask the analyst to tune, which is the entire contrast with `k` in KNNimpute and `λ` in the penalized methods.

So let me write it as the algorithm I'd actually run, then as code. The skeleton:

  initial fill: every hole ← column mean (numeric) / modal class (categorical)
  order ← variables sorted by increasing number of missing values
  Δ_old ← ∞
  repeat (up to max_iter sweeps):
    X_old ← current imputed matrix
    for s in order, for variables that have missing values:
      fit a random forest: observed(X_s) ~ other columns of current matrix at observed-s rows
      predict the missing X_s from other columns at missing-s rows
      write predictions into the matrix
    X_new ← current imputed matrix
    Δ_new ← Σ(X_new − X_old)² / Σ X_new²    (relative squared change)
    if Δ_new > Δ_old:  return X_old           (the iterate just before the rise — the minimum)
    Δ_old ← Δ_new
  return current matrix                         (only reached if the cap fires first)

Now I can fill the one empty slot in the round-robin harness — the per-variable predictor and the convergence/return logic. For continuous tables the predictor is a `RandomForestRegressor` and the convergence test is the continuous Δ; with factor columns I use the same loop with modal-class initialization, a classifier for factor responses, and the PFC-style Δ_F.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor


class CustomImputer(BaseEstimator, TransformerMixin):
    """MissForest: iterative random-forest imputation.

    Round-robin (chained-equations) skeleton with a random forest as the
    per-variable predictor. Start from a complete mean fill; sweep the
    variables in increasing order of missingness, each time fitting a forest
    on the observed part of a variable against the current values of the other
    variables and overwriting that variable's holes with the forest's
    predictions; stop when the sweep-to-sweep change in the imputed matrix
    first increases (return the iterate just before the rise), else at the cap.
    """

    def __init__(self, random_state=42, max_iter=10, n_estimators=100):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = n_estimators

    def fit(self, X, y=None):
        # learn the imputation from X only (the observed entries); no test labels
        self._impute(X)
        return self

    def transform(self, X):
        return self._impute(X)

    def fit_transform(self, X, y=None):
        return self._impute(X)

    def _impute(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        missing = np.isnan(X)                       # original missingness pattern

        # initial complete fill: column means, so every predictor block is
        # defined from the first sweep (the best no-covariate constant guess)
        col_mean = np.nanmean(X, axis=0)
        X_imp = X.copy()
        for j in range(n_features):
            X_imp[missing[:, j], j] = col_mean[j]

        # variables that actually need imputing, sorted by ascending missingness:
        # the most-complete variable trains the cleanest forest first, and its
        # improved column then helps the harder variables later in the same sweep
        miss_count = missing.sum(axis=0)
        order = [j for j in np.argsort(miss_count) if miss_count[j] > 0]
        if not order:
            return X_imp

        mtry = max(1, int(np.floor(np.sqrt(n_features))))  # canonical m_try = floor(sqrt(p))
        X_best = X_imp.copy()                        # iterate to return
        delta_old = np.inf
        for _ in range(self.max_iter):
            X_prev = X_imp.copy()

            for j in order:                          # Gauss-Seidel sweep
                obs = ~missing[:, j]
                mis = missing[:, j]
                others = [k for k in range(n_features) if k != j]
                if not others:
                    X_imp[mis, j] = np.nanmean(X[obs, j])
                    continue

                X_train = X_imp[obs][:, others]      # current fill of the other columns
                y_train = X[obs, j]                  # genuinely observed targets only
                X_pred = X_imp[mis][:, others]

                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,  # ~100: error has flattened, runtime ~linear
                    max_features=min(mtry, len(others)),
                    min_samples_leaf=5,               # R randomForest regression nodesize default
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                X_imp[mis, j] = rf.predict(X_pred)    # overwrite this variable's holes

            # relative squared change of the imputed matrix between sweeps
            denom = np.sum(X_imp ** 2)
            delta_new = np.sum((X_imp - X_prev) ** 2) / denom if denom > 0 else 0.0

            # stop the first time the change turns back up: the previous sweep
            # sat at the minimum, before the forest noise started churning
            if delta_new > delta_old:
                return X_best
            X_best = X_imp.copy()
            delta_old = delta_new

        return X_best                                 # cap reached: return the latest
```

If I want a production harness rather than this explicit loop, scikit-learn's `IterativeImputer` gives me the same chained-equations shell: initial mean fill, ascending feature order, a per-feature `estimator.fit/predict`, and a `max_iter` cap. I should not pretend it is identical, though. Its built-in early stop is `max(abs(X_t - X_{t-1})) / max(abs(X[known_vals])) < tol`, and it returns the latest iterate; the original rule watches the relative squared change, stops when that change rises, and returns the previous iterate. So `IterativeImputer(estimator=RandomForestRegressor(...), initial_strategy="mean", imputation_order="ascending")` is the right sklearn skeleton, while the exact stop-return-previous behavior needs the custom loop above.

I end where the requirements forced me to go. Mean imputation ignores the rest of the row, so it cannot recover anything structural — I need to *predict* each hole from the other variables. Neighborhood prediction is continuous-only, needs a tuned `k`, needs standardization, and blurs nonlinearities. Model-based chained equations handle mixed types via per-variable conditional regressions iterated round-robin — the right skeleton — but make me specify a parametric model per variable and assume an implied joint distribution. The skeleton is keepable; the per-variable model is what's wrong. The spec sheet for the predictor I need — mixed-type, scale-free, nonlinear, no parametric family, no tuning, robust — is exactly the spec sheet of a random forest, so I plug a forest into each slot. To make "regress on the other variables" well-defined despite holes in those variables too, I keep a complete working matrix and refine it; within each sweep I take variables in ascending-missingness order so the best-trained forests run first and improve the inputs for the rest. Breiman's classification and regression bounds both point at the same choice, low tree correlation without throwing away too much strength, so `m_try = floor(sqrt(p))`; the convergence of forest error as trees are added makes about 100 trees a precision/runtime knee rather than a tuning burden. The forest's out-of-bag rows, about 36.8% per tree, give a per-variable and hence per-type imputation-error estimate without any held-out set or cross-validation. And the iteration stops by reading its own dynamics — the sweep-to-sweep change descends, then turns up when the imputation enters the forest-noise jitter, so I return the iterate at that minimum — which means the whole method carries no tuning parameter the analyst has to set.

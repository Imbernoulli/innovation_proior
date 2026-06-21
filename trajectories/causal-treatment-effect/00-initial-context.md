## Research question

Estimate the **conditional average treatment effect** $\tau(x)=E[Y(1)-Y(0)\mid X=x]$ from observational data $(X,T,Y)$ — covariates, a binary treatment, and a single observed outcome — where treatment assignment depends on $X$ through an unknown mechanism. The task is to implement the estimator that fills `fit(X, T, Y)` / `predict(X)`; the data-generating processes, cross-fitting, and metrics are fixed.

Three obstacles define the problem. **Confounding:** the raw treated-minus-control difference mixes the treatment effect with differences between the kinds of units that receive treatment. **Heterogeneity:** the effect varies across $X$, so the target is a function, not a single number. **Missing counterfactuals:** for each unit only one potential outcome is observed, so $Y(1)-Y(0)$ is never directly available and there is no held-out label for $\tau(x)$. Identification relies on unconfoundedness, $\{Y(0),Y(1)\}\perp T\mid X$, and overlap, $0<e(x)<1$ for $e(x)=P(T=1\mid X)$, both built into the DGPs.

## Prior art / Background / Baselines

The known estimators are meta-learners and related approaches that turn off-the-shelf supervised learners into CATE estimators.

- **Naive difference of conditional means.** Fit a separate regression of $Y$ on $X$ in each treatment arm and subtract the two fitted surfaces.
- **S-learner and T-learner.** The S-learner pools both arms and includes treatment as an extra feature; the T-learner fits one model per arm and subtracts.
- **Inverse-propensity weighting.** Estimate the propensity score $e(x)=P(T=1\mid X)$ and reweight each observed outcome by $1/e(x)$ for treated units and $1/(1-e(x))$ for controls.
- **Orthogonalization / double machine learning.** First residualize $Y$ and $T$ on $X$, then estimate $\tau$ from the residual relationship.

## Fixed substrate / Code framework

A frozen synthetic benchmark generates $(X,T,Y)$ together with ground-truth $\tau$ and ATE. The DGPs are:

- `ihdp_synth`: IHDP-flavored, $n=747$, $p=25$, nonlinear effects.
- `jobs_synth`: Jobs/LaLonde-flavored, $n=2000$, $p=10$, earnings-scale outcomes.
- `acic_synth`: ACIC-flavored, $n=4000$, $p=50$, correlated high-dimensional confounding.

Each DGP constructs a nonlinear propensity with interactions and quadratics, clips it to $[0.05,0.95]$ to enforce overlap, draws $T\sim\mathrm{Bernoulli}(e(x))$, and builds outcomes from complex nonlinear response surfaces. Evaluation is 5-fold cross-fitting repeated over 10 data seeds; the estimator is `deepcopy`-cloned fresh per fold. `scikit-learn`, `numpy`, and `scipy` are available, along with `BaseCATEEstimator` and the metric utilities `compute_pehe` and `compute_ate_error`.

## Editable interface

Only the `CATEEstimator` class in `custom_cate.py` (lines 344–416) is editable. It must satisfy:

- `fit(self, X, T, Y) -> self`: learn from $(n,p)$ covariates `X`, $(n,)$ binary treatment `T`, and $(n,)$ outcome `Y`.
- `predict(self, X)`: return $(n,)$ per-row estimates $\hat\tau(x)$.

The default fill is a placeholder S-learner: one `Ridge` trained on $[X \mid T]$, read off by toggling $T$.

```python
# EDITABLE region of custom_cate.py (lines 344-416) — default fill (placeholder S-Learner)
class CATEEstimator(BaseCATEEstimator):
    """Custom CATE estimator. Default: a simple S-Learner placeholder (Ridge)."""

    def __init__(self):
        pass

    def fit(self, X, T, Y):
        # Placeholder: simple S-Learner (augmented features)
        n, p = X.shape
        XT = np.column_stack([X, T.reshape(-1, 1)])
        self._model = Ridge(alpha=1.0)
        self._model.fit(XT, Y)
        return self

    def predict(self, X):
        n = X.shape[0]
        X1 = np.column_stack([X, np.ones((n, 1))])
        X0 = np.column_stack([X, np.zeros((n, 1))])
        return self._model.predict(X1) - self._model.predict(X0)
```

## Evaluation settings

Each DGP is scored with 5-fold cross-fitting over 10 data seeds (the leaderboard fixes the launch seed at 42). Both metrics are lower-is-better:

- **PEHE** (Precision in Estimation of Heterogeneous Effects) $=\sqrt{\operatorname{mean}((\hat\tau-\tau)^2)}$ — pointwise heterogeneity error.
- **ATE error** $=\lvert\operatorname{mean}(\hat\tau)-\mathrm{ATE}_{\text{true}}\rvert$ — aggregate bias.

Because results are averaged across folds and seeds, the estimator must be stable across train/test splits.

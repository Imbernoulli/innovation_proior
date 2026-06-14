## Research question

Estimate the **conditional average treatment effect** $\tau(x)=E[Y(1)-Y(0)\mid X=x]$ from
observational data $(X,T,Y)$ — covariates, a binary treatment, a single observed outcome — where
treatment was *not* assigned by me but by some unknown mechanism out in the world, so the people who
got treated differ systematically in $X$ from those who did not. The single thing being designed is
the estimator that fills `fit(X, T, Y)` / `predict(X)`. Everything else — the synthetic
data-generating processes, the cross-fitting evaluation, the metrics — is fixed.

Three facts box the problem in and never go away. **Confounding:** the naive treated-minus-control
gap mixes the treatment's effect with the effect of *being the kind of unit that gets treated*.
**Heterogeneity:** the effect varies across the covariate space in nonlinear, interaction-laden ways,
so a single number is not the target — a whole function is. **The fundamental obstruction:** for any
one unit I see exactly one of $Y(1),Y(0)$, never both, so the per-unit contrast $Y(1)-Y(0)$ is not in
the data at all and there is no held-out label to score $\tau$ directly. Identification rests on two
assumptions the DGPs are built to satisfy: **unconfoundedness** $\{Y(0),Y(1)\}\perp T\mid X$ and
**overlap** $0<e(x)<1$ for the propensity $e(x)=P(T=1\mid X)$.

## Prior art before the first rung (the meta-learner lineage the ladder reacts to)

The ladder is a sequence of *meta-learners* — recipes that turn an off-the-shelf supervised learner
into a CATE estimator by how they slice the data and recombine the fits, never by inventing a bespoke
causal loss inside the learner. The lineage the first rung reacts to:

- **The naive difference of conditional means.** Under unconfoundedness $E[Y\mid X=x,T=w]=\mu_w(x)$, so
  $\tau(x)=\mu_1(x)-\mu_0(x)$ is a difference of two ordinary regressions. The whole literature is
  organized around *which* regressions to fit and *how* to debias them. Gap: stated this way it puts
  the entire burden on getting two high-dimensional response surfaces right, and offers no defense when
  they are wrong.
- **Outcome-modeling meta-learners (S-/T-learner; Künzel et al. 2019).** Pool the data with $T$ as a
  feature and toggle it (S), or fit one model per arm and subtract (T). Gap: the S-learner lets a
  regularized learner shrink the weak treatment feature toward zero (effect attenuated); the T-learner
  estimates two surfaces in isolation and inherits the *response*-surface rate even when $\tau$ is far
  simpler, with the two independent errors corrupting the difference.
- **Propensity reweighting (Horvitz & Thompson 1952; Rosenbaum & Rubin 1983).** Collapse the confounding
  into the scalar balancing score $e(x)$ and invert it: weight treated outcomes by $1/e$, controls by
  $1/(1-e)$, so the confounding cancels in expectation with *no* outcome model. Gap: dividing by $e(1-e)$
  makes the variance explode exactly where overlap is weak — a single near-zero $\hat e$ dominates.
- **Orthogonalization / double machine learning (Robinson 1988; Chernozhukov et al. 2018; Nie & Wager
  2021; Kennedy 2023; Athey & Wager 2018).** Residualize $Y$ and $T$ on $X$ before estimating $\tau$,
  so the bias of $\hat\tau$ becomes a *product* of nuisance errors (doubly robust / Neyman-orthogonal)
  and the effect can converge at its own complexity rather than the nuisances'. This is the modern
  frontier the strong rungs land in; the ladder climbs toward it from the weak, un-orthogonalized end.

## The fixed substrate

A synthetic-benchmark harness is frozen and must not be touched. Three task-local DGP families generate
$(X,T,Y)$ together with the ground-truth $\tau$ and ATE: **`ihdp_synth`** (IHDP-flavored, $n=747$,
$p=25$, nonlinear effects), **`jobs_synth`** (Jobs/LaLonde-flavored, $n=2000$, $p=10$, earnings-scale
outcomes), **`acic_synth`** (ACIC-flavored, $n=4000$, $p=50$, high-dimensional correlated confounding).
Each DGP builds a nonlinear propensity with interactions and quadratics, clips $e$ to $[0.05,0.95]$ to
enforce overlap, draws $T\sim\mathrm{Bernoulli}(e)$, and forms outcomes from complex nonlinear response
surfaces. Evaluation is **5-fold cross-fitting** (fit on $K-1$ folds, predict the held-out fold,
aggregate the out-of-fold $\hat\tau$) repeated over **10 data seeds**; the estimator is `deepcopy`-cloned
fresh per fold. `scikit-learn`, `numpy`, and `scipy` are available; the abstract `BaseCATEEstimator`
(with `fit`/`predict`) and the metric utilities (`compute_pehe`, `compute_ate_error`) are provided.

## The editable interface

Exactly one region is editable — the `CATEEstimator` class in `custom_cate.py` (lines 344–416). Every
method on the ladder is a fill of this same two-method contract:

- `fit(self, X, T, Y) -> self`: learn from the $(n,p)$ covariates `X`, the $(n,)$ binary treatment `T`,
  and the $(n,)$ outcome `Y`.
- `predict(self, X)`: return the $(n,)$ per-row treatment-effect estimates $\hat\tau(x)$.

The starting point is the scaffold default: a **placeholder S-learner** — one `Ridge` on $[X\mid T]$,
read off by toggling $T$. Each later method replaces exactly this class and nothing else.

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

Each DGP is scored over 5-fold cross-fitting × 10 data seeds (the leaderboard rows fix the launch seed
at 42). Two metrics, **both lower is better**:

- **PEHE** (Precision in Estimation of Heterogeneous Effects) $=\sqrt{\operatorname{mean}((\hat\tau-\tau)^2)}$
  — the primary, pointwise, heterogeneity-sensitive metric.
- **ATE error** $=\lvert\operatorname{mean}(\hat\tau)-\mathrm{ATE}_{\text{true}}\rvert$ — the
  aggregate-bias metric.

Because the estimator is cross-fit and averaged over many data seeds, it must be *stable across
train/test splits*, not tuned to one realization.

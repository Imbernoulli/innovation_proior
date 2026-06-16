# BOHB, distilled

BOHB (Bayesian Optimization and HyperBand) is a hyperparameter optimizer that keeps
Hyperband's multi-fidelity resource-allocation skeleton intact and replaces only its random
sampling of configurations with a model-based proposal. The model is a TPE-style density ratio:
fit two kernel density estimators per budget — one on the best configurations, one on the rest —
and propose the configuration that maximizes the good/bad density ratio (provably the
expected-improvement maximizer). This pairs Hyperband's strong anytime performance with
Bayesian optimization's strong final performance, while staying simple, scalable to mixed
high-dimensional spaces, parallelizable, and robust.

## Problem it solves

Find `x* = argmin_{x in X} f(x)` where `f` is a model's validation loss over a mixed
(continuous / integer / categorical) configuration space, observed only noisily, with each
evaluation enormously expensive. Required: good configurations after a budget of only a few
full trainings (anytime), the best configuration given more budget (final), near-linear use of
parallel workers, scalability to many hyperparameters, and robustness across hyperparameter
types — none of which any single prior method delivers at once.

## Key idea

Hyperband is good at *allocating budget* (its bracket/successive-halving schedule) and bad at
*choosing configurations* (it samples them uniformly at random and never learns). A model-based
acquisition is the reverse. So:

- **Keep Hyperband's schedule unchanged.** Brackets `s = s_max ... 0`, geometric budgets
  `b_max·eta^{-s}`, successive halving (keep the top `1/eta`, multiply budget by `eta`), with
  ideal bracket sizes `n_ideal=((s_max+1)/(s+1))·eta^s` so
  `(s+1)·n_ideal·b_max·eta^{-s}=(s_max+1)·b_max` before integer rounding.
- **Replace the random configuration draw with a density-ratio model.** Keep observations
  separated by budget; per budget, fit `l(x)` (a KDE on the best configurations) and `g(x)`
  (a KDE on the rest); propose `argmax l(x)/g(x) = argmin g(x)/l(x)`.

**Why `argmin g/l` is EI.** With threshold `alpha` at quantile `gamma` of observed losses
(`gamma = p(y < alpha)`), `l(x) = p(x | y < alpha)`, `g(x) = p(x | y >= alpha)`, and marginal
`p(x) = gamma·l(x) + (1-gamma)·g(x)`,

```
EI(x) = ∫_{-∞}^{α} (α - y) p(y|x) dy
      = l(x)·[∫_{-∞}^{α}(α-y)p(y)dy] / p(x)
      ∝ ( gamma + (1-gamma)·g(x)/l(x) )^{-1},
```

monotone decreasing in `g(x)/l(x)`. So maximizing EI ⟺ minimizing `g/l`. No model of `f`,
`p(y)`, or the `x`-free constant is needed.

## Design choices and why

- **KDE / TPE, not a Gaussian process.** GP fitting is `O(|D|^3)` and the bandit skeleton makes
  *many* cheap low-budget evaluations, so a cubic model's cost would dominate; GPs also need
  bespoke kernels for mixed/categorical spaces and careful hyperpriors. A KDE is linear in
  `|D|`, native to mixed spaces, and simple.
- **Model the largest budget with enough data.** The objective is `f = tilde f(·, b_max)`, so
  use the highest-fidelity model that is trustworthy. Cheap fidelities bootstrap the search
  early; as evidence accrues on larger budgets, the model used climbs the fidelity ladder,
  which lets BOHB overturn wrong low-budget conclusions.
- **`N_min = d + 1`.** A `d`-dimensional KDE needs at least `d+1` points; otherwise degenerate.
- **Good/bad split with nondegenerate KDE checks.** Request
  `n_good = max(N_min, floor(top_n_percent·N_b/100))` best configurations and
  `n_bad = max(N_min, floor((100-top_n_percent)·N_b/100))` worse configurations for the bad
  density, then fit only when the realized good and bad slices both have more rows than
  dimensions. Default `top_n_percent = 15`.
- **Optimize EI by sampling, not full optimization.** Draw `N_s = 64` candidates and take the
  best ratio. Not fully optimizing keeps consecutive proposals diverse → near-linear parallel
  speedups.
- **Sample candidates from a *widened* good KDE (`bandwidth × bw`, `bw = 3`).** The
  largest-budget model is queried often but updated rarely, so sampling from the tight fitted
  density stagnates; widening restores exploration around promising configs and diversifies
  parallel proposals. Score with the *un-widened* ratio (widening only affects *where to look*).
- **Random fraction `rho = 1/3`.** Preserves Hyperband's convergence guarantee. In the worst
  case of misleading lower fidelities, BOHB is at most `rho^{-1}·(s_max+1)` times slower than
  random search in full-budget random-evaluation accounting, and still converges.
- **Single multivariate KDE (product of 1-D *kernels*), not TPE's product of 1-D *pdfs*.**
  Captures hyperparameter interactions.
- **Scott's-rule (`normal_reference`) bandwidths**, floored at `min_bandwidth = 1e-3` (a
  collapsed dimension would otherwise become a non-exploring spike); **`eta = 3`**; crashed /
  non-finite losses counted as `+inf` (bad).

## Final algorithm (sampling a configuration)

```
on each completed evaluation at budget b:
    store loss, using +inf for crashed or non-finite runs
    if no larger budget already has a model:
        sort D_b by loss and request the good/bad split above
        fit l and g only if both realized slices are nondegenerate

when sampling a configuration:
    if rand() < rho or no KDE pair exists: return a uniform random configuration
    b = largest budget with a fitted KDE pair
    draw N_s candidates from l' (= l with all bandwidths * bw)
    return the candidate minimizing g(x)/l(x)
```

The Hyperband scheduler decides which budget each proposed configuration runs at. Observations
are shared and accumulated across all brackets and iterations.

## Working code

The model-based config generator (the part that replaces Hyperband's random sampling),
mirroring the canonical implementation:

```python
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm


class BOHBConfigGenerator:
    """Fits, per budget, good/bad multivariate KDEs on the best/worst split of the
    observed configurations, and proposes argmin g(x)/l(x) (= EI maximizer).
    Replaces the random configuration sampling inside Hyperband."""

    def __init__(self, space, top_n_percent=15, num_samples=64, random_fraction=1/3,
                 bandwidth_factor=3, min_bandwidth=1e-3, min_points_in_model=None, seed=42):
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        d = space.dim
        self.min_points = (d + 1) if min_points_in_model is None else max(min_points_in_model, d + 1)
        self.kde_vartypes = "".join('u' if p.type == 'categorical' else 'c' for p in space.params)
        self.vartypes = np.array([len(p.choices) if p.type == 'categorical' else 0
                                  for p in space.params], dtype=int)
        self.configs = {}        # budget -> list of encoded config vectors
        self.losses = {}         # budget -> list of losses (lower is better)
        self.kde_models = {}     # budget -> {'good': KDE, 'bad': KDE}

    def new_result(self, config, loss, budget):
        loss = loss if np.isfinite(loss) else np.inf         # crashed / non-finite -> bad
        self.configs.setdefault(budget, []).append(self.space.to_array(config))
        self.losses.setdefault(budget, []).append(loss)

        if self.kde_models and max(self.kde_models) > budget:  # a larger budget already modeled
            return
        if len(self.configs[budget]) <= self.min_points - 1:
            return

        X = np.array(self.configs[budget]); y = np.array(self.losses[budget])
        n_good = max(self.min_points, (self.top_n_percent * X.shape[0]) // 100)
        n_bad = max(self.min_points, ((100 - self.top_n_percent) * X.shape[0]) // 100)
        order = np.argsort(y)
        good = X[order[:n_good]]; bad = X[order[n_good:n_good + n_bad]]
        if good.shape[0] <= good.shape[1] or bad.shape[0] <= bad.shape[1]:
            return
        good_kde = sm.nonparametric.KDEMultivariate(good, var_type=self.kde_vartypes, bw='normal_reference')
        bad_kde = sm.nonparametric.KDEMultivariate(bad, var_type=self.kde_vartypes, bw='normal_reference')
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)
        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        self.kde_models[budget] = {'good': good_kde, 'bad': bad_kde}

    def get_config(self, budget=None):
        # rho fraction random, and random until a model exists
        if not self.kde_models or self.rng.rand() < self.random_fraction:
            return self.space.sample_uniform(self.rng)

        b = max(self.kde_models)                              # largest budget with a model
        kde_good = self.kde_models[b]['good']; kde_bad = self.kde_models[b]['bad']
        l = kde_good.pdf; g = kde_bad.pdf
        ratio = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)  # EI is monotone decr. in g/l

        best_val, best_vec = np.inf, None
        for _ in range(self.num_samples):
            datum = kde_good.data[self.rng.randint(len(kde_good.data))]
            vec = []
            for value, bw, vt in zip(datum, kde_good.bw, self.vartypes):
                bw = max(bw, self.min_bandwidth)
                if vt == 0:                                    # continuous: widened jitter
                    bw = self.bw_factor * bw
                    a, b_ = (0 - value) / bw, (1 - value) / bw
                    vec.append(sps.truncnorm.rvs(a, b_, loc=value, scale=bw, random_state=self.rng))
                else:                                          # categorical: keep, or flip
                    if self.rng.rand() < (1 - bw):
                        vec.append(int(value))
                    else:
                        vec.append(self.rng.randint(vt))
            val = ratio(vec)
            if val < best_val:
                best_val, best_vec = val, vec
        if best_vec is None:
            return self.space.sample_uniform(self.rng)
        return self.space.from_array(best_vec)
```

Hyperband's bracket schedule (unchanged), which decides how many configurations at which budget:

```python
import numpy as np


def hyperband_brackets(min_budget, max_budget, eta=3):
    """Geometric ladder of successive-halving brackets; each bracket costs ~ the same
    total resource. Bracket s starts n0 configs at budget max_budget*eta^{-s}."""
    max_sh_iter = int(np.log(max_budget / min_budget) / np.log(eta)) + 1
    budgets = max_budget * eta ** (-np.arange(max_sh_iter - 1, -1, -1))
    brackets = []
    for s in range(max_sh_iter - 1, -1, -1):
        n0 = int(np.floor(max_sh_iter / (s + 1)) * eta ** s)          # integer approximation
        ns = [max(int(n0 * eta ** (-i)), 1) for i in range(s + 1)]    # configs surviving each rung
        brackets.append({'budgets': budgets[-(s + 1):], 'num_configs': ns})
    return brackets
```

Default hyperparameters: `eta = 3`, `top_n_percent = 15`, `num_samples = 64`,
`random_fraction = 1/3`, `bandwidth_factor = 3`, `min_bandwidth = 1e-3`,
`min_points_in_model = d + 1`.

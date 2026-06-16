# Tree-structured Parzen Estimator (TPE)

TPE is a sequential model-based optimizer for expensive black-box hyper-parameter search. It
fills the SMBO template with an unusual surrogate: instead of modeling the loss given a
configuration, `p(y|x)`, it models the configuration given the loss, `p(x|y)`. It splits the
observed trials at a quantile of their loss into a "good" set and a "bad" set, fits a
non-parametric (Parzen/KDE) density to each — `l(x)` over the good configurations, `g(x)` over
the bad — and proposes the configuration that maximizes Expected Improvement, which the algebra
reduces to maximizing the density ratio `l(x)/g(x)`. It is cheap (per iteration linear in the
history and in the dimension), needs no metric on the input, and handles continuous, discrete,
categorical, and conditional (tree-structured) variables natively.

## Problem it solves

Find a low-loss (high-score) configuration in as few expensive trials as possible over a
mixed-type, conditionally structured hyper-parameter space, using the information from every
past trial — concentrating on promising regions without prematurely collapsing onto one basin,
at a per-round cost far below the cost of one model fit.

## Key idea

Within the SMBO loop (fit a cheap surrogate to the history `H`; maximize an acquisition over
it to pick the next `x*`; evaluate; refit), use Expected Improvement as the acquisition,
`EI_{y*}(x) = E[max(y* - Y(x), 0)]`, but flip the surrogate. Model `p(x|y)` with two densities
split at a threshold `y*`:

```
p(x|y) = l(x)  if y < y*        (l: density of configs among the GOOD trials)
         g(x)  if y >= y*       (g: density of configs among the BAD trials)
```

Choose `y*` as the `gamma`-quantile of the observed losses, `gamma = p(y < y*)`, so that a
fraction `gamma` of points fall below it and `l` has data to be built from (this is why `y*` is
a quantile, not the best-so-far, which would leave `l` empty).

## The EI collapse (why l/g)

```
EI_{y*}(x) = int_{-inf}^{y*} (y* - y) p(y|x) dy
           = int_{-inf}^{y*} (y* - y) p(x|y) p(y) / p(x) dy.
```

On `(-inf, y*)`, `p(x|y) = l(x)` (no `y`-dependence), so it factors out:

```
numerator  = l(x) * int_{-inf}^{y*} (y* - y) p(y) dy
           = l(x) * ( gamma*y* - int_{-inf}^{y*} y p(y) dy ).
p(x)       = int p(x|y) p(y) dy = gamma*l(x) + (1 - gamma)*g(x).
```

Divide top and bottom by `l(x)`: the top is a constant in `x` (it contains only `y*` and the
marginal `p(y)`), so

```
EI_{y*}(x)  ∝  ( gamma + (1 - gamma) * g(x)/l(x) )^{-1}.
```

EI is strictly decreasing in `g(x)/l(x)`, so **maximizing EI = maximizing l(x)/g(x)**. The
marginal `p(y)` drops out of the ordering entirely — it is never modeled. Acquisition is
trivial: draw candidate configurations from `l(x)` (already concentrated where good points
are) and return the one with the largest `l(x)/g(x)` (equivalently `log l(x) - log g(x)`); the
ratio penalizes regions that are also dense in bad trials.

## Why it beats GP-EI in this regime

- No `O(n^3)` GP solve; densities are sorted once and scored in `O(n)` per candidate.
- No metric needed on `x`: a Parzen density uses per-coordinate kernels and composes onto the
  generative tree (one density per node), so categorical/conditional variables are native.
- Acquisition is sample-and-rank, not a multi-modal global search over `x`.
- Exploration is structural, not staked on a fragile predictive standard error: GP-EI is a
  two-stage method whose only exploration term is the estimated `s(x)`, which a sparse/deceptive
  initial sample can collapse to zero (then `EI = 0` everywhere and the search stalls). TPE
  instead folds the prior into each density as an extra mixture component, so probability never
  drops to zero anywhere in the legal range.

## The adaptive Parzen densities

- **Continuous variable** with prior on `(a, b)` (uniform / Gaussian / log-uniform): `l` is an
  equally-weighted mixture of {the prior} and {one Gaussian per good observation `x^(i)`}. The
  prior component is the exploration floor.
- **Bandwidths** are adaptive: after the prior center is inserted among the observed centers,
  each non-prior Gaussian's sigma is set from the larger left/right neighbor gap, then clipped to
  `[prior_sigma/min(100, 1+N_components), prior_sigma]`; the prior component keeps
  `prior_sigma`. The upper clip prevents over-smoothing past the legal range, and the lower clip
  prevents a single-point spike.
- **Log-uniform variables** are handled in the log domain.
- **Categorical variable** with prior probabilities `p_i`: posterior weight of choice `i`
  proportional to `K*p_i + C_i`, where `K` is the number of choices and `C_i` counts choice
  `i` in the good set — the `K*p_i` prior count keeps every choice explorable (the categorical
  analogue of the prior-mixture floor).
- **`g`** is built identically over the bad set.

## Defaults and why

`gamma = 0.25`, `n_ei_candidates = 24`, prior mixture weight `= 1.0`, `n_startup = 10` for
this harness, and a linear-forgetting window of about `25` in the canonical implementation.

- `gamma` is the exploit/explore dial *and* the guarantee that `l` has data: small `gamma` =>
  `l` from few elite points (sharp, exploitative, noisy); large `gamma` => `l` broad
  (exploratory). Default `0.25` (an even more selective `0.15` is reasonable when the surrogate
  is trusted to concentrate).
- `n_ei_candidates = 24`: enough draws from `l` to find a high-ratio one without paying for many
  density evaluations; keeps the per-round cost well below one model fit.
- `n_startup`: KDE on a handful of points is noise, so seed with random search (which is itself
  sampling the prior `l`/`g` are built on) until the densities are estimable.
- Practical refinements that do not change the EI derivation: let the good set grow only like
  `ceil(gamma * sqrt(n))` so it stays elite as the history lengthens; linearly down-weight the
  oldest observations so the densities track where the search has moved.

## Algorithm

```
seed H with n_startup random configurations drawn from the prior
loop until budget exhausted:
    n_good <- max(1, floor(gamma * len(H)))
    y* <- np.sort(scores)[-n_good]      # score threshold; higher is better
    split H: good = trials with score >= y*  ->  build l(x);   bad = the rest  ->  build g(x)
    draw n_EI_candidates configurations from l(x)
    return argmax over candidates of  log l(x) - log g(x)      # = argmax l/g = argmax EI
    evaluate, append to H
```

## Score-orientation note

The derivation above minimizes a loss `y`. When the objective is a validation *score* (higher
is better), "good" means *high* score: set `n_good = max(1, int(gamma * len(history)))`, set
`threshold = np.sort(scores)[-n_good]`, put trials with score `>= threshold` into the good set
that builds `l`, and the rest into `g`. The acquisition is unchanged — still
`argmax (log l - log g)`.

## Working code

Filling the `suggest()` slot of the sequential HPO harness, with an explicit Gaussian-kernel
Parzen estimate so the `l/g` ratio is transparent:

```python
import numpy as np
from math import erf, pi, sqrt
from typing import Any, Dict, List, Tuple


class CustomHPOStrategy:
    """Tree-structured Parzen Estimator (TPE).

    Models p(x|y) by splitting the history at a quantile into a 'good' density
    l(x) and a 'bad' density g(x); proposes the candidate maximizing EI, which
    reduces to maximizing l(x)/g(x).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.25          # quantile split: keep the best fraction as 'good' (the l-set)
        self.n_startup = 10        # random search until densities are estimable
        self.n_ei_candidates = 24  # candidates drawn from l(x) and ranked by l/g each round
        self.prior_weight = 1.0

    def _bounds(self, p):
        low, high = float(p.low), float(p.high)
        if p.log_scale:
            low, high = np.log(low), np.log(high)
        return low, high

    def _to_numeric(self, p, value):
        value = float(value)
        return np.log(value) if p.log_scale else value

    def _from_numeric(self, p, value):
        if p.log_scale:
            value = np.exp(value)
        if p.type == "int":
            value = int(round(value))
        return value

    def _normal_cdf(self, x, mu, sigma):
        return 0.5 * (1.0 + erf((x - mu) / (sqrt(2.0) * max(float(sigma), 1e-12))))

    def _logsumexp(self, values):
        values = np.asarray(values, dtype=float)
        m = np.max(values)
        return float(m + np.log(np.sum(np.exp(values - m))))

    def _categorical_probs(self, p, trials):
        choices = list(p.choices)
        prior = getattr(p, "prior", None)
        prior = np.ones(len(choices)) if prior is None else np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        counts = np.zeros(len(choices), dtype=float)
        for t in trials:
            counts[choices.index(t.config[p.name])] += 1.0
        weights = counts + self.prior_weight * len(choices) * prior
        return weights / weights.sum()

    def _parzen_components(self, p, trials):
        low, high = self._bounds(p)
        prior_mu = 0.5 * (low + high)
        prior_sigma = max(high - low, 1e-12)
        obs = np.array([self._to_numeric(p, t.config[p.name]) for t in trials], dtype=float)

        if len(obs) == 0:
            mus = np.array([prior_mu])
            sigmas = np.array([prior_sigma])
            prior_pos = 0
        elif len(obs) == 1:
            if obs[0] < prior_mu:
                mus = np.array([obs[0], prior_mu])
                sigmas = np.array([0.5 * prior_sigma, prior_sigma])
                prior_pos = 1
            else:
                mus = np.array([prior_mu, obs[0]])
                sigmas = np.array([prior_sigma, 0.5 * prior_sigma])
                prior_pos = 0
        else:
            obs = np.sort(obs)
            prior_pos = int(np.searchsorted(obs, prior_mu))
            mus = np.insert(obs, prior_pos, prior_mu)
            sigmas = np.empty_like(mus)
            sigmas[0] = mus[1] - mus[0]
            sigmas[-1] = mus[-1] - mus[-2]
            sigmas[1:-1] = np.maximum(mus[1:-1] - mus[:-2], mus[2:] - mus[1:-1])

        minsigma = prior_sigma / min(100.0, 1.0 + len(mus))
        sigmas = np.clip(sigmas, minsigma, prior_sigma)
        sigmas[prior_pos] = prior_sigma

        weights = np.ones(len(mus), dtype=float)
        weights[prior_pos] = self.prior_weight
        weights /= weights.sum()
        return weights, mus, sigmas

    def _sample_param(self, p, trials):
        if p.type == "categorical":
            choices = list(p.choices)
            probs = self._categorical_probs(p, trials)
            return choices[int(self.rng.choice(len(choices), p=probs))]

        low, high = self._bounds(p)
        weights, mus, sigmas = self._parzen_components(p, trials)
        while True:
            k = int(self.rng.choice(len(mus), p=weights))
            value = self.rng.normal(mus[k], sigmas[k])
            if low <= value <= high:
                break
        return self._from_numeric(p, value)

    def _logpdf_param(self, p, value, trials):
        if p.type == "categorical":
            choices = list(p.choices)
            probs = self._categorical_probs(p, trials)
            return float(np.log(probs[choices.index(value)] + 1e-30))

        low, high = self._bounds(p)
        x = self._to_numeric(p, value)
        weights, mus, sigmas = self._parzen_components(p, trials)
        accepts = np.array([
            self._normal_cdf(high, mu, sigma) - self._normal_cdf(low, mu, sigma)
            for mu, sigma in zip(mus, sigmas)
        ])
        p_accept = float(np.sum(weights * accepts))
        terms = []
        for w, mu, sigma in zip(weights, mus, sigmas):
            log_coef = np.log(w + 1e-30) - np.log(max(p_accept, 1e-30))
            log_coef -= np.log(sqrt(2.0 * pi) * max(float(sigma), 1e-12))
            terms.append(log_coef - 0.5 * ((x - mu) / max(float(sigma), 1e-12)) ** 2)
        return self._logsumexp(terms)

    def _sample_from_density(self, space, trials):
        config = {p.name: self._sample_param(p, trials) for p in space.params}
        return space.clip(config)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        # startup: sample the prior (= random search) until the densities are worth estimating
        if len(history) < self.n_startup:
            return space.sample_uniform(self.rng), 1.0

        # split at the score quantile: 'good' = top gamma (higher score is better),
        # threshold = the n_good-th best score; defines y* implicitly.
        scores = np.array([t.score for t in history])
        n_good = max(1, int(self.gamma * len(history)))
        threshold = np.sort(scores)[-n_good]

        good_trials = [t for t in history if t.score >= threshold]  # l(x)
        bad_trials = [t for t in history if t.score < threshold]    # g(x)
        if len(bad_trials) == 0:
            bad_trials = good_trials

        # propose-and-rank: draw from l(x), keep greatest log l(x) - log g(x)
        best_score = -np.inf
        best_config = None
        for _ in range(self.n_ei_candidates):
            candidate = self._sample_from_density(space, good_trials)
            log_l = sum(self._logpdf_param(p, candidate[p.name], good_trials) for p in space.params)
            log_g = sum(self._logpdf_param(p, candidate[p.name], bad_trials) for p in space.params)
            ei = log_l - log_g                            # maximize l/g  <=>  maximize EI
            if ei > best_score:
                best_score = ei
                best_config = candidate

        return best_config, 1.0
```

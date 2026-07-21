The problem in front of me is hyper-parameter search for an expensive model — a deep belief net with thirty-odd knobs whose validation loss depends sharply on getting them right. Each evaluation means training the whole model and scoring it, an hour of GPU time, so my entire budget is at most a couple hundred trials. The objective is a black box: a single scalar back, no gradient, no view inside. And the space is hostile in a way that rules out most off-the-shelf optimizers: a learning rate is continuous, a hidden-unit count is ordinal, a pre-processing choice is categorical, and the space is *conditional* — "units in layer two" only exists if "number of layers" came out at least two — so any method has to choose values *and* decide which variables are even live for a given point. The cheapest honest description of such a space is itself a generative procedure: draw the number of layers from its prior, then conditioned on that draw each layer's parameters, on down the tree. Random search is exactly "run that procedure, keep the best," and it is a serious reference point, not a strawman, because the response function has low effective dimensionality — only a few of the thirty knobs matter on any data set, and which few differ across data sets — so $S$ random draws give every axis $S$ distinct values and cover the axes that matter far better than a grid, which burns its budget on the cartesian product. But random search has one glaring waste: it is stateless. Every draw ignores every loss already paid for. With only two hundred trials, throwing away the history is the thing I cannot afford.

The standard cure is Sequential Model-Based Optimization: keep a cheap surrogate of the loss built from the history $H$, and each round maximize an acquisition criterion over that surrogate to pick the next point $x^\*$, pay to evaluate $f(x^\*)$, append, and refit. For the acquisition I want Expected Improvement, because it balances exploitation against exploration with no hand-set target: given the best loss $y^\*$ seen so far and a surrogate that hands me a predictive distribution $p_M(y\mid x)$ over the unknown loss at $x$,
$$\mathrm{EI}_{y^\*}(x) = \mathbb{E}\!\left[\max(y^\* - Y(x),\, 0)\right] = \int_{-\infty}^{\infty} \max(y^\* - y,\, 0)\; p_M(y\mid x)\, dy.$$
The usual surrogate slotted in here is a Gaussian process, and it is elegant — closed under conditioning, returning an analytic posterior mean $\hat y(x)$ and standard deviation $s(x)$, so that EI has the closed form $s(x)\,[u\,\Phi(u) + \phi(u)]$ with $u = (y^\* - \hat y(x))/s(x)$. But stress-tested against this regime it fails on four fronts. Conditioning a GP on $n$ points costs $O(n^3)$. Its kernel presumes a metric on the whole configuration vector, which is not even well-defined when different points have different active variables — forcing a hand-built tree of independent GPs stitched together. Maximizing EI over $x$ becomes a hard multi-modal global search in more than ten dimensions with no gradient in the categorical directions. And — the one that actually changes my mind — GP-EI is a two-stage procedure that stakes all of its exploration on the single estimated standard error $s(x)$; if a sparse, deceptive early sample makes the fit under-estimate $s(x)$ (the textbook sine sampled exactly at its crests, where $s(x)=0$ everywhere so $\mathrm{EI}=0$ everywhere), exploration silently dies and the search stalls. With a couple hundred trials in dozens of dimensions, a deceptive early sample is my *normal* starting condition, not an edge case.

I propose the Tree-structured Parzen Estimator. The starting question is what EI actually needs: not a calibrated regressor on $x$, only an *ordering* over candidates, since I draw candidates and rank them by EI. The GP factors the joint as $p(y\mid x)\,p(x)$ and models the hard direction; my data is naturally the other direction — for each trial I have its configuration and its loss, samples of the joint. So I factor it the other way, $p(x\mid y)\,p(y)$, and model $p(x\mid y)$, the distribution over *configurations* given the loss, which I can estimate by simply grouping observed configurations by outcome and fitting a density to each group — and a density needs only per-coordinate kernels, never a global metric. The simplest non-trivial version splits the loss at a single threshold $y^\*$:
$$p(x\mid y) = \begin{cases} \ell(x) & \text{if } y < y^\* \quad(\text{density of configs among the GOOD trials}),\\ g(x) & \text{if } y \ge y^\* \quad(\text{density of configs among the BAD trials}).\end{cases}$$
Crucially $y^\*$ cannot be the best-so-far the way GP-EI uses it, because then $\{y < y^\*\}$ would be empty and $\ell$ would have no data. Instead I set $y^\*$ as the $\gamma$-quantile of the observed losses, $\gamma = p(y < y^\*)$, so a fraction $\gamma$ of points fall below it by construction and $\ell$ always has data to be built from.

Pushing this through the EI integral is where the construction earns itself. Substituting Bayes, $p(y\mid x) = p(x\mid y)\,p(y)/p(x)$, and noting $\max(y^\*-y,0)$ kills everything above $y^\*$,
$$\mathrm{EI}_{y^\*}(x) = \int_{-\infty}^{y^\*} (y^\* - y)\,\frac{p(x\mid y)\,p(y)}{p(x)}\, dy.$$
On $(-\infty, y^\*)$, $p(x\mid y) = \ell(x)$ has no $y$-dependence, so it factors straight out, and using $\int_{-\infty}^{y^\*} p(y)\,dy = \gamma$ the numerator becomes $\ell(x)\,(\gamma y^\* - \int_{-\infty}^{y^\*} y\,p(y)\,dy)$. The denominator by total probability is $p(x) = \int p(x\mid y)\,p(y)\,dy = \gamma\,\ell(x) + (1-\gamma)\,g(x)$. Dividing top and bottom by $\ell(x)$, the entire top reduces to a factor containing only $y^\*$ and the marginal $p(y)$ — no $x$ at all — leaving
$$\mathrm{EI}_{y^\*}(x) \;\propto\; \left(\gamma + (1-\gamma)\,\frac{g(x)}{\ell(x)}\right)^{-1}.$$
The marginal $p(y)$ — the very piece I feared I had discarded by going to a binary split — has dropped out of the ordering entirely, surviving only in a constant prefactor that is the same for every candidate. EI is strictly decreasing in $g(x)/\ell(x)$, so **maximizing Expected Improvement is exactly maximizing $\ell(x)/g(x)$** — go where good points are likely and bad points are not. That intuitive statement fell out of the algebra rather than being assumed, and it makes the acquisition step trivial: $\ell$ is just the generative process re-weighted toward good observations, so I sample candidates *from* $\ell$ (already concentrated where good points live), then rank them by $\log \ell(x) - \log g(x)$, the ratio penalizing any candidate that sits in a region that is also dense in bad trials. No gradient, no restarts, no global optimizer — just sample-and-rank, at a per-round cost linear in the history and the dimension, with categoricals and conditionals handled natively because the density only needs per-coordinate kernels on the generative tree.

The parameter $\gamma$ does real work: it is the exploit-versus-explore dial (small $\gamma$ builds $\ell$ from a few elite points — sharp, exploitative, noisy; large $\gamma$ builds it from a broad swath — exploratory) and simultaneously the guarantee that $\ell$ has data. A default of $\gamma = 0.25$ keeps the best quarter as good; $0.15$ is reasonable when the surrogate is trusted to concentrate. Building $\ell$ and $g$ is where I make exploration robust where the GP could not. For a continuous variable with prior on $(a,b)$, the Parzen estimate is an equally-weighted mixture of {the original prior} together with {one Gaussian per good observation}. A pure mixture of the data would be near-zero anywhere unsampled and would collapse onto its own history — the same excessive-locality death, by a different mechanism — so folding the prior in as one more component gives a probability *floor* everywhere in the legal range. That floor is my exploration mechanism, and it is structural: it does not depend on estimating any quantity that could collapse to zero the way $s(x)$ did. Each Gaussian's bandwidth is adaptive rather than global: after inserting the prior center among the sorted observed centers, each non-prior $\sigma$ is set from the larger of its left/right neighbor gaps (dense clusters get narrow kernels, lonely points wide ones), then clipped into $[\,\text{prior\_sigma}/\min(100, 1+N), \text{prior\_sigma}\,]$ — the upper clip stops over-smoothing past the legal range, the lower clip stops a single-point spike from earning a runaway EI — while the prior component keeps the full prior width. Log-uniform variables are handled entirely in the log domain. Categoricals get the same data-plus-prior idea: the posterior weight of choice $i$ is proportional to $K\,p_i + C_i$, where $p_i$ is the prior probability, $K$ the number of choices, and $C_i$ the count of choice $i$ in the good set — the $K\,p_i$ prior count keeps every choice explorable, the categorical analogue of the prior-mixture floor. Two further refinements sharpen the densities without touching the derivation — letting the good set grow only like $\lceil \gamma\sqrt{n}\,\rceil$ so it stays genuinely elite as the history lengthens, and linearly down-weighting the oldest observations so the densities track where the search has moved — but they earn their keep over campaigns much longer than the couple hundred trials I actually have here, so the version below keeps the plain $\gamma n$ split and equal weighting instead. A short random-search startup ($n_{\text{startup}} = 10$) seeds the densities, since KDE on a handful of points is noise — and random search *is* sampling the substrate that $\ell$ and $g$ are built on — and I draw $24$ EI candidates per round, enough to find a high-ratio one cheaply. One arithmetic note for the harness: it reports a validation *score* where higher is better, so "good" means high score — with $n_{\text{good}} = \max(1, \lfloor \gamma n \rfloor)$ the threshold is the $n_{\text{good}}$-th best score, trials at or above it build $\ell$ and the rest build $g$, and everything downstream is unchanged.

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

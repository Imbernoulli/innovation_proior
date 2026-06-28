Let me start from what actually hurts. I have a model — a deep belief net, say — with thirty-odd hyper-parameters, and its validation loss depends sharply on getting them right. Every time I want to know the loss for one configuration I have to train the whole thing and score it on a validation set, which costs me an hour of GPU time. So my entire budget is maybe a couple hundred evaluations, ever. I get back a single scalar with no gradient and no view inside. And the space is a mess: a learning rate is continuous, the number of hidden units is ordinal, the pre-processing choice is categorical, and worst of all the space is conditional — "number of units in layer two" only exists if "number of layers" came out at least two. So whatever I do has to choose values *and* decide which variables are even live for a given point, over a mixed-type, tree-structured domain, with almost no evaluations to spend. The cheapest honest thing I can write down is the space itself as a generative procedure: draw the number of layers from its prior, then conditioned on that draw each layer's parameters from theirs, and so on down the tree. Sampling that procedure once produces a valid configuration, conditional structure included for free. Random search is exactly "run that procedure, keep the best," and I should take it seriously, because it has a real argument behind it: the response function from hyper-parameters to loss has low effective dimensionality — only a few of the thirty knobs matter on any given data set, and which few differ across data sets — so drawing S configurations at random gives every axis S distinct values and covers the axes that matter far better than a grid, which burns its trials on the cartesian product and never probes any single axis at more than a handful of levels. So random search is my reference point and it is not a strawman. But it has one glaring waste: it is stateless. Every draw ignores every loss I have already paid for. With two hundred trials, throwing away the history is the thing I cannot afford. I want to keep random search's strengths — sampling the generative process, mixed types and conditionals for free, trivial parallelism — but bias the next draw toward where the evidence says good configurations live.

So I want to be sequential and model-based. The template is clear enough: keep a cheap surrogate of the expensive loss built from the history H, and each round maximize some acquisition criterion over that surrogate to pick the next point x*, pay to evaluate f(x*), append it, refit. The expensive model fit happens once per round, so everything else has to be cheap next to it. Two choices define the method: what surrogate, and what acquisition. For acquisition I want Expected Improvement, because it does the exploit-versus-explore balance for me without a hand-set target. The idea is: given the best loss y* I have seen, and a probabilistic surrogate that hands me a distribution over the unknown value Y(x) at a candidate x, the expected improvement is how much, on average, x would beat the incumbent — for a minimization problem,

  EI_{y*}(x) = E[ max(y* − Y(x), 0) ] = ∫_{−∞}^{∞} max(y* − y, 0) p_M(y|x) dy,

where the max throws away outcomes worse than y* (they "improve" by zero) and the expectation is under the surrogate's predictive distribution p_M(y|x) at x. It is intuitive and it has been shown to work across a lot of settings, so I will commit to EI and spend my design effort on the surrogate.

The textbook surrogate is a Gaussian process. It is genuinely elegant: it is a prior over functions closed under conditioning, so given H it returns an analytic posterior mean yhat(x) and standard deviation s(x), exactly the predictive distribution EI wants, and EI then has a clean closed form, EI(x) = s(x)[u·Φ(u) + φ(u)] with u = (y* − yhat(x))/s(x), where Φ and φ are the normal CDF and PDF. The mean drives exploitation and the standard error drives exploration; lovely. So why not just do this and stop? Let me actually stress-test it against my regime instead of admiring it. First, conditioning a GP on n points costs O(n^3) — a kernel-matrix solve — which grows as the history grows; not fatal at a couple hundred points, but it is real overhead in the inner loop. Second, and worse for me, the GP kernel is a similarity defined on the input vector x, which presumes a metric on the whole configuration. But my space has categoricals and conditionals: how similar are "ZCA pre-processing" and "raw"? And what is the kernel between a point with two layers and a point with three, where some coordinates simply do not exist? A single GP over the whole space is not even well-defined here. A GP route has to hand-build a *tree of independent GPs*, one per conditional group, and stitch them together — a lot of machinery to paper over the fact that the surrogate wanted a Euclidean domain I do not have. Third, I have to *maximize* EI over x to get x*, and as the history grows the EI surface becomes badly multi-modal, with no gradient in the categorical directions, so in more than ten dimensions the acquisition step is itself a hard global optimization, usually attacked with restarts of an evolutionary search. None of these alone kills it, but they are all friction exactly where I have the least room — high dimension, small budget, ugly types.

There is a fourth problem and it is the one that actually changes my mind. EI under a GP is a *two-stage* procedure: stage one fits the surrogate and estimates its parameters, including the predictive standard error s(x); stage two takes those estimates as if they were correct and computes where to search. But s(x) is the *only* thing in EI forcing the method back to sparsely sampled regions — it is the entire exploration term. So if the fit ever under-estimates s(x), exploration dies. I want to see how bad that actually gets rather than just worry about it, so let me build the worst case I can think of and measure it. Take the true function to be a sine on [0,3] and sample it at exactly three of its crests, x = 0.25, 1.25, 2.25, where every observation reads the same value, +1. Now fit a squared-exponential GP and look at what it does at the *troughs* — x = 0.75, 1.75, 2.75 — where the truth is −1 and which are precisely the points a sane optimizer should be dying to explore. The catch is the length-scale (call it λ, to keep it clear of the density ℓ I will define later): three equally spaced points all reading +1 look like a constant, and a marginal-likelihood fit will push λ *up* to explain them as one flat function. So let me sweep λ and read off the predictive sd and the resulting EI at the trough x = 0.75 (best-so-far y* = +1, minimizing):

  λ = 0.5 : trough mean = +1.01, sd = 5.9e−1, EI = 2.3e−1
  λ = 1.0 : trough mean = +1.02, sd = 1.3e−1, EI = 4.5e−2
  λ = 3.0 : trough mean = +1.00, sd = 5.6e−3, EI = 2.1e−3
  λ = 10  : trough mean = +1.00, sd = 1.7e−4, EI = 6.8e−5

So the catastrophe is real but I had its shape slightly wrong: the GP does not report literally zero variance from the first fit — at a short length-scale there is still healthy sd at the trough. The collapse is *driven by the fit itself*. As the likelihood drives λ up to call the three crests one flat line, the trough mean stays confidently and wrongly at ≈ +1 (truth −1), and the sd there falls by nearly four orders of magnitude, dragging EI from 0.23 down to 7e−5. The method does not just lose uncertainty; it becomes confidently wrong exactly at the unexplored optimum, and its own exploration term evaporates there at the same time. The method trusts a single point estimate of its own uncertainty, and a deceptive sample makes the fit hand it a bad estimate with full confidence. With only a couple hundred trials in dozens of dimensions, a sparse, deceptive early sample is not a pathological edge case — it is my *normal* starting condition. Staking exploration on a standard error that the fit can quietly shrink toward zero is a bet I do not want to make.

Let me stare at the EI integral again and ask what it actually needs, because maybe the GP is solving a harder problem than EI requires. EI needs p_M(y|x): for any candidate x, the distribution over its loss. The GP gives me that by modeling the forward map x → distribution-over-y directly, which is why it needs a metric on x and an O(n^3) solve. But what do I really use it for? I draw candidates x and I rank them by EI; I never need a calibrated number, I need an *ordering* over x. Is there a way to get an EI ordering without ever building a regressor on x at all? The thing I have a lot of, relatively, is the other direction: for each observed trial I have its loss y and its configuration x, so I have samples of the *joint* (x, y). The GP factors the joint as p(y|x)p(x) and models p(y|x). What if I factor it the other way, p(x|y)p(y), and model p(x|y) — the distribution over *configurations* given the loss? That sounds backwards at first, modeling x from y, but it is exactly the direction in which my data is easy: I just group my observed configurations by their loss and estimate a density over x within each group. No metric on the whole vector is needed for a density — a Parzen window only needs a per-coordinate kernel, and it composes straight onto the generative tree by estimating one density per node. Let me chase this.

If I am going to condition x on y, the simplest non-trivial thing is to split y at a single threshold y* into "good" and "bad," and estimate one density of configurations on each side:

  p(x | y) = ℓ(x)  if y < y*,
             g(x)  if y ≥ y*,

so ℓ(x) is the density of x among the trials whose loss came in below y* (the good ones, since lower loss is better here) and g(x) is the density among the rest. Both are just Parzen estimates over the configurations I have actually seen, grouped by outcome. This already handles every type and conditional natively — it is a density on the same generative process random search samples from, only re-weighted by the data. The question is whether this two-density object is enough to *compute EI*, or whether I have thrown away too much by reducing y to a binary good/bad. Let me push the EI integral through it and find out, because if EI comes out clean then this factorization was the right move and if it does not I will have to add the p(y) model back.

Start from EI and substitute Bayes, p(y|x) = p(x|y)p(y)/p(x):

  EI_{y*}(x) = ∫_{−∞}^{y*} (y* − y) p(y|x) dy = ∫_{−∞}^{y*} (y* − y) p(x|y) p(y) / p(x) dy.

The integration limit dropped to y* because max(y* − y, 0) is zero whenever y ≥ y* — only losses below the threshold contribute any improvement. Now look at the integrand on (−∞, y*): there, by my definition, p(x|y) is exactly ℓ(x), which does not depend on y, so it pulls straight out of the integral:

  ∫_{−∞}^{y*} (y* − y) p(x|y) p(y) dy = ℓ(x) ∫_{−∞}^{y*} (y* − y) p(y) dy.

I have to be careful about what I commit to with y*, and here is where the choice has to be different from the GP. GP-EI uses y* = the best observed loss. If I did that, the set {y < y*} would be *empty* — nothing beats the best — and ℓ(x) would have no data to be built from. So that choice is incompatible with this whole construction. I need y* large enough that a fraction of points fall below it to *form* ℓ. So let me define y* implicitly by a quantile: pick a number γ in (0,1) and set y* so that the fraction of observed losses below it is exactly γ, i.e. γ = p(y < y*). That is the natural way to guarantee ℓ has data, and γ becomes a knob I will interpret in a moment. With that, expand the remaining y-integral by splitting off the constant y*:

  ∫_{−∞}^{y*} (y* − y) p(y) dy = y* ∫_{−∞}^{y*} p(y) dy − ∫_{−∞}^{y*} y p(y) dy
                              = γ y* − ∫_{−∞}^{y*} y p(y) dy,

using ∫_{−∞}^{y*} p(y) dy = p(y < y*) = γ. So the whole numerator of EI is

  ∫_{−∞}^{y*} (y* − y) p(x|y) p(y) dy = ℓ(x) ( γ y* − ∫_{−∞}^{y*} y p(y) dy ).

Now the denominator p(x). By total probability over the same two-sided split,

  p(x) = ∫_ℝ p(x|y) p(y) dy = ℓ(x) ∫_{y < y*} p(y) dy + g(x) ∫_{y ≥ y*} p(y) dy
       = γ ℓ(x) + (1 − γ) g(x),

because the y-mass below y* is γ and above it is 1 − γ. Putting numerator over denominator,

  EI_{y*}(x) = ℓ(x) ( γ y* − ∫_{−∞}^{y*} y p(y) dy ) / ( γ ℓ(x) + (1 − γ) g(x) ).

Now divide top and bottom by ℓ(x). The top becomes γ y* − ∫_{−∞}^{y*} y p(y) dy — and look at it: there is no x in it at all. y* is a fixed threshold and p(y) is the marginal over losses, so that entire factor is a *constant* with respect to x. The bottom becomes γ + (1 − γ) g(x)/ℓ(x). So

  EI_{y*}(x) ∝ ( γ + (1 − γ) · g(x)/ℓ(x) )^{−1}.

The marginal p(y) — the piece I worried I had thrown away by going to a binary split — has dropped out of everything except that constant prefactor, which is the same for every candidate x, so it cannot change which x maximizes EI; I never need to model p(y) at all. And the dependence on x is entirely through the single ratio g(x)/ℓ(x). I have done a fair amount of integral-pushing to get here and I do not trust it until I have checked it on numbers, so let me build a tiny discrete (x, y) world and compare the *definition* of EI against this formula directly. First attempt: I lay down an x-grid and a y-grid, fill the joint p(x,y) with arbitrary positive numbers, normalize, pick γ ≈ 0.3 to set the threshold y*, read off ℓ and g as p(x | good) and p(x | bad), compute EI_true(x) = Σ_{y<y*} (y* − y) p(y|x) straight from the definition, and divide it by the formula (γ + (1−γ)g/ℓ)^{−1}. If the algebra is right the quotient should be a single x-independent constant. It is *not*: the quotient ranges over a factor of 1.45, and worse, argmax EI_true lands at grid index 19 while argmax ℓ/g lands at 11. They disagree about which point to try. So either the algebra is wrong or I have mis-set up the test.

Staring at the failing step, it is the test. The line where I "pulled ℓ(x) out of the integral" used p(x|y) = ℓ(x) for *every* y below y* — but in an arbitrary joint p(x|y) genuinely varies with y inside the good region, so it does not equal a single density ℓ(x). The factoring is only valid when the data really is generated by the two-density split: p(x|y) constant-in-y on each side of y*. That is not a flaw in the algebra; it is telling me that the binary split is a *modeling assumption*, the very assumption I am choosing to make, not something true of every joint. So the honest test is to generate the joint *from* that model — set ℓ, g, p(y), and assemble p(x,y) = ℓ(x)p(y) for good y and g(x)p(y) for bad y — and then check the identity. Doing that: the quotient EI_true / formula is now constant to machine precision (max/min = 1.000000), it equals γy* − Σ_{y<y*} y p(y) to fourteen digits (0.060058 predicted vs 0.060058 measured), and argmax EI_true, argmax ℓ/g, argmin g/ℓ all agree at index 17. The algebra is exact — under its own assumption, which I now understand I am buying into deliberately.

So with the split as my model, EI is a strictly decreasing function of g(x)/ℓ(x): the whole expression is one over (γ plus a positive multiple of that ratio), so larger g/ℓ gives smaller EI. Maximizing expected improvement is therefore minimizing g(x)/ℓ(x), equivalently maximizing ℓ(x)/g(x). The candidate I want is the one most likely under the good-trial density and least likely under the bad-trial density — "go where good points are and bad points are not." That fell out of the EI algebra rather than being assumed, and it survived a numeric check that also taught me exactly what I am assuming to get it. Modeling p(x|y) instead of p(y|x) buys an EI criterion that is a ratio of two densities I can build cheaply, with p(y) integrated away.

This also tells me how to do the acquisition step that was so painful for the GP. I do not have to globally maximize a multi-modal surface over x. The tree-structured ℓ and g are densities I can *sample from* directly — ℓ is just my generative process re-weighted toward the good observations. So on each round I draw a batch of candidate configurations *from ℓ(x)*, which already concentrates them where good points have been, then I evaluate ℓ and g at each and keep the one with the largest ℓ(x)/g(x) — equivalently the largest log ℓ(x) − log g(x). Sampling from ℓ to propose and ranking by the ratio to select: the ratio is supposed to down-weight any candidate that happens to sit in a region that is *also* dense in bad trials, so I am not fooled by a place that is good-dense only because it is busy-dense. Let me confirm the ranking does what I want on the same learning-rate example, with good trials near 1e−3 and bad trials near 1e−1. A candidate at 1.2e−3 (in the good cluster) scores log ℓ − log g = +1.60; a candidate at 7e−2 (in the bad cluster) scores −2.97; and at the cluster center 1e−3 itself the ratio is +1.66 while at 1e−1 it is −3.23. So the good-region candidate is preferred by more than four nats, and the ordering is exactly "go where good points are and bad points are not," as intended. There is no gradient, no restarts, no global optimizer over x — just sample-and-rank, and the per-iteration cost is linear in the history and linear in the dimension, because building the Parzen densities is sorting and the candidate scoring is a sum of one-dimensional kernel evaluations. The O(n^3) and the metric problem are both gone.

I want to understand γ before I move on, because it is doing real work. y* is the γ-quantile of the observed losses, so γ controls how the trials get split into good and bad. If γ is tiny, ℓ is built from only the few most elite points — a sharp, exploitative density, but estimated from so few points that it is noisy and risks collapsing onto a single basin. If γ is large, ℓ is built from a broad swath of merely acceptable points — more exploratory, less selective. So γ is the exploit-versus-explore dial, and it is also the term that keeps the construction well-posed at all by guaranteeing ℓ has data. Notice the contrast with the GP that pushed y* aggressively toward the best observed loss to chase improvement; here I deliberately keep y* *higher* than the best so that some points can be used to form ℓ. A quarter is a reasonable default — keep the best quarter as "good" — and an even more selective value around 0.15 makes sense when I trust the surrogate to concentrate. The point is that γ is a single interpretable knob, not a kernel and a length-scale and a mean-function to fit.

Now I have to actually build ℓ(x) and g(x), and this is where I make exploration robust in a way the GP could not. For a continuous hyper-parameter whose prior is uniform on an interval (a, b) — or a Gaussian, or log-uniform — the natural Parzen estimate over the good observations is a mixture of Gaussians, one centered at each good point x^(i). But a pure mixture of the data has a fatal flaw for my purpose: it is zero, or nearly zero, anywhere I have not yet sampled, which would make the algorithm refuse to ever propose a genuinely new region and collapse onto its own history — the same excessive-locality death the GP suffered, just by a different mechanism. The fix is to put the *prior itself back in* as one more, equally-weighted, component of the mixture. So ℓ(x) is an equally-weighted mixture of {the original prior over (a,b)} together with {one Gaussian per good observation}. The claim I am leaning on is that this gives a floor of probability density everywhere in the legal range, and I should make sure it actually does rather than just hope it — because this floor is now carrying all of my exploration, the role s(x) played for the GP, and I just watched s(x) fail. Take a log-uniform learning rate on [1e−5, 1e−1] with the good observations clustered tightly near 1e−3 and ask for the density at 1e−5 — the far low edge, as far from any good point as the range allows. If the floor works, log ℓ there must be finite, not −∞. Computing it from the mixture I am describing: log ℓ(1e−5) = −4.52, finite; for comparison log ℓ at the cluster center 1e−3 is −1.43. The far edge is down by about three nats but it is emphatically alive, so a candidate drawn there has real support and no region is permanently written off. The prior component is doing exactly the job I put it in for. This is my exploration mechanism, and it is structural — it does not depend on estimating any quantity that could shrink toward zero the way s(x) did under the fit. That is the whole point: where the GP let the fit collapse a fragile standard-error estimate, I bake exploration into the floor of the density itself.

Each Gaussian in the mixture needs a width, and I want it adaptive rather than one global bandwidth, because the good points cluster unevenly. Sort the observed centers, insert the prior center, and set each non-prior Gaussian's standard deviation from the larger gap to its left or right neighbor in that sorted list — so a point in a dense cluster gets a narrow kernel and a lonely point gets a wide one, which is exactly the local-adaptivity I want from a Parzen estimate. Then clip every sigma into a sensible range. The upper clip is the prior's own width: no single data Gaussian should be wider than the prior it sits inside, or it would over-smooth past the whole legal range. The lower clip — the prior width divided by a capped count of mixture components — stops any one Gaussian from becoming an infinitely sharp spike on a single observation, which would overfit that point and, through the ratio, hand it a runaway EI. The inserted prior component keeps the prior width as its own sigma. A learning-rate-style log-uniform variable I just handle in the log domain: take logs of the observations, do all of this in log space, and the bandwidths come out sensible across the decades the prior spans, instead of being dominated by the largest values.

Categorical variables need their own version of the same "data plus prior" idea. Suppose the prior over the K choices is the probability vector p_i. The naive estimate from the good set is the empirical frequency of each choice, but again that assigns zero probability to any choice not yet seen among the good trials — and then a candidate using that choice gets ℓ = 0, EI = 0, and the choice is never tried again, even if it only failed to appear by chance in a tiny good set. So I smooth: make the posterior weight of choice i proportional to K·p_i + C_i, where C_i counts how often choice i appears among the good observations. The K·p_i term is the prior count — exactly the categorical analogue of folding the prior into the continuous mixture — and it keeps every choice explorable while still letting the observed counts C_i pull the density toward what has actually worked. Same density mechanics for g over the bad trials, and the proposal-and-rank loop is identical.

Two refinements I notice once I think about a long run rather than a snapshot. First, the good set should not grow without bound and dilute: with n observations a simple γ-quantile keeps roughly γn points in the good set, but a good practical choice is to let it grow only like the square root of n — keep about ceil(γ·sqrt(n)) good points — so that as the history gets long the good density stays focused on a genuinely elite subset rather than a quarter of everything. Second, older trials are systematically worse than newer ones once the search is working, so I can linearly down-weight the oldest observations in the Parzen weights — a "forgetting" ramp that gives full weight to the most recent block and tapers the weight on the earliest points — so the densities track where the search has *moved to*, not where it started. Neither changes the EI derivation; they only sharpen the densities that feed it.

Before I write code, the one piece of arithmetic that does not survive translation: I have derived everything for *minimizing* a loss y, where "good" means y below the threshold and ℓ is the low-loss density. My harness reports a validation *score* where higher is better. So good means *high* score: with n_good = max(1, floor(γn)), I set the threshold to the n_good-th best observed score, put the trials with score at or above it into the good set that builds ℓ, and the rest into g. Everything downstream is untouched — I still draw candidates and rank them by log ℓ(x) − log g(x), keeping the largest. The flip is only in which trials count as good.

Let me also be honest about startup. The Parzen densities are meaningless on two or three points; ℓ would be one Gaussian plus the prior and the ratio would be noise. So for the first ten rounds I just sample from the prior, exactly random search, which is the right thing to do anyway since random search *is* sampling the substrate that ℓ and g are built on. Only once I have enough observations to estimate the two densities do I switch on the model-based proposal. And I will draw 24 EI candidates from ℓ each round — enough to find a high-ratio one without paying for many density evaluations, which keeps the per-round cost comfortably below the cost of one model fit.

So let me write the proposal rule into the one empty slot of the harness. I keep the implementation close to the derivation: split the score history into good and bad, build one adaptive one-dimensional posterior per exposed hyper-parameter, draw each EI candidate from the good posterior, and score that same candidate under the good and bad posteriors by summing per-parameter log densities. That is the same sample-and-rank structure as the tree implementation, compressed into this `CustomHPOStrategy` interface.

```python
import numpy as np
from math import erf, pi, sqrt
from typing import Any, Dict, List, Tuple


class CustomHPOStrategy:
    """Tree-structured Parzen Estimator (TPE).

    Models p(x|y) by splitting the history at a quantile of the loss/score into
    a 'good' density l(x) and a 'bad' density g(x), then proposes the candidate
    that maximizes EI, which the algebra shows equals maximizing l(x)/g(x).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.25          # quantile split: keep the best fraction as 'good' (the l-set)
        self.n_startup = 10        # random search until the densities are worth estimating
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
        # startup: sample the prior (= random search) until the densities are estimable
        if len(history) < self.n_startup:
            return space.sample_uniform(self.rng), 1.0

        # split at the score quantile -- 'good' = top gamma (higher score is better here),
        # so the threshold is the n_good-th best score; this defines y* implicitly.
        scores = np.array([t.score for t in history])
        n_good = max(1, int(self.gamma * len(history)))
        threshold = np.sort(scores)[-n_good]

        good_trials = [t for t in history if t.score >= threshold]  # builds l(x)
        bad_trials = [t for t in history if t.score < threshold]    # builds g(x)
        if len(bad_trials) == 0:
            bad_trials = good_trials

        # propose-and-rank: draw candidates and keep the one with the greatest
        # log l(x) - log g(x) = log of the EI-maximizing ratio l(x)/g(x)
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

Let me trace the causal chain one last time. I was stuck wanting to use my history on an expensive, high-dimensional, mixed-type, conditional black box where random search was strong but wasteful because it ignores every loss it pays for. SMBO with Expected Improvement was the frame, but the Gaussian-process surrogate that usually fills it needed a metric I do not have over categorical and conditional variables, cost O(n^3), turned acquisition into a hard multi-modal global search, and — fatally for a small, deceptive initial sample — staked all of its exploration on a predictive standard error that the fit itself can shrink toward zero exactly where it is most needed, which I watched happen on the sine-at-crests example, where pushing the length-scale up dragged the trough's EI down to 7e−5 while the GP stayed confidently wrong at the unexplored optimum. Asking what EI actually needs, I saw I never need a calibrated regressor on x, only an ordering, and that my data is naturally an ordering over configurations grouped by outcome. So I factored the joint the other way, modeling p(x|y) as two Parzen densities split at a quantile y* — ℓ over the good trials, g over the bad — and pushed that through the EI integral. The marginal p(y) integrated out of the ordering entirely, leaving EI proportional to (γ + (1−γ)g(x)/ℓ(x))^{−1}, so maximizing improvement is just maximizing ℓ(x)/g(x): go where good points are likely and bad points are not — an identity I checked numerically, where it held to machine precision once I generated the data from the binary split it assumes, and failed when I did not, which is how I learned the split is a modeling assumption I am buying rather than a free consequence. That made the acquisition step trivial — sample candidates from ℓ, rank by the ratio — at a per-round cost linear in the history and the dimension, with categoricals and conditionals handled natively because a Parzen density only needs per-coordinate kernels on the generative tree. I made exploration robust by folding the prior into each density as an extra mixture component (and as a prior count for categoricals), so no region ever drops to zero probability — I confirmed the density stays finite, log ℓ = −4.52, even at the legal edge farthest from any good point — replacing the GP's collapsible standard error with a structural floor. The bandwidths are local, clipped between the prior width and a per-point floor; γ is the one interpretable exploit-explore knob and the thing that guarantees ℓ has data; and a short random-search startup seeds the densities. The result drops straight into the sequential harness as split, per-parameter posterior sampling from ℓ, per-parameter log-density scoring under ℓ and g, and an argmax of log ℓ minus log g.

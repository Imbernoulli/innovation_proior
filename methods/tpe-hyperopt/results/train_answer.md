The problem I am facing is hyper-parameter optimization for deep models — a deep belief network, a stacked denoising autoencoder, a convolutional net — where a single configuration carries anywhere from ten to fifty knobs: a classifier learning rate, an annealing schedule, an L2 penalty, a pre-processing choice, a number of layers, and then, *per layer*, a hidden-unit count, a weight-init scheme, contrastive-divergence epochs and learning rate and annealing. Each trial is one full configuration turned into a trained, evaluated model, costing real wall-clock, so my whole budget is tens to maybe a couple hundred trials. Two features make this not a vanilla box-optimization problem. The variables are mixed-type — continuous (a learning rate), ordinal (hidden units), categorical (raw vs ZCA pre-processing). And the space is *conditional*: "number of units in the second layer" is only a meaningful variable when "number of layers" came out $\geq 2$, so a configuration is not a fixed-length vector — the very set of active variables depends on choices made higher up the tree. The optimizer must decide *which* variables to optimize at the same time as their values.

The obvious tools each break in an instructive way. Grid search is exponential in the number of hyper-parameters and burns resolution on axes the loss barely cares about — dead on arrival at fifty dimensions. Manual tuning, what people actually do, is genuinely good when only a handful of trials are affordable, but it does not scale, is unreproducible, and cannot systematically exploit a cluster. Random search is the real baseline, and it has a beautiful property: the clean way to even describe this conditional, mixed-type space is as a *generative process* — draw the number of layers; conditioned on that, draw each existing layer's parameters; draw the pre-processing; and so on — and a valid configuration is exactly a draw of that process, so "which variables are active" falls out of the draw for free. Random search is then just sample-from-prior, evaluate, repeat, and it is known to beat grid search because it never wastes trials on insensitive coordinates. Its fatal crack is that it is *memoryless*: it draws from the same prior every time and never looks at the losses it has already measured, so on the hard datasets it converges slowly or plateaus below careful search. Every trial is gold and the information is thrown away. What I want is to use the history to steer the next draw while keeping random search's effortless handling of the tree. The natural frame is sequential model-based optimization: replace the expensive $f(x)$ with a cheap surrogate fit to the history $H = \{(x_i, f(x_i))\}$, at each step optimize a cheap criterion to pick the next $x$, evaluate the true $f$ there (the one expensive call), append, refit. The criterion is the settled part — Expected Improvement, which given a threshold $y^\ast$ and a model giving a distribution over the value at $x$ scores $x$ by $$\mathrm{EI}_{y^\ast}(x) = \int_{-\infty}^{y^\ast} (y^\ast - y)\, p_M(y \mid x)\, dy,$$ weighting each possible gain by both its size and its probability so it trades exploitation against exploration with no hand-set knob. The textbook surrogate is a Gaussian process over $p(y \mid x)$, but bolting it onto this problem strains badly: the GP fit is $O(|H|^3)$; EI on a GP is a multimodal surface that must be globally optimized over a ten-to-fifty-dimensional mixed space, demanding a heavy auxiliary optimizer (EDA on categoricals, CMA-ES on the continuous part, restarts) wrapped around an opaque criterion; and worst, the conditional tree does not fit one fixed-dimensional GP kernel, forcing a hand-carve into groups with an independent GP per group. The thing random search did for free — respect the generative tree — the GP fights me on.

I propose the Tree-structured Parzen Estimator, TPE. The decisive move is to flip the modeling direction: instead of $p(y \mid x)$, model the *generative* direction $p(x \mid y)$ together with the marginal $p(y)$. This feels backwards until I notice that $p(x \mid y)$ is a distribution over *configurations*, and from random search I already own an entire apparatus for representing and sampling distributions over configurations — the generative process — so if I express $p(x \mid y)$ in that same shape the tree structure is respected automatically. To define $p(x \mid y)$ from a pile of $(x_i, y_i)$, the cleanest operation is to *threshold* the losses: pick $y^\ast$ and split, modeling the conditional as two densities, $$p(x \mid y) = l(x) \ \text{if}\ y < y^\ast, \qquad g(x) \ \text{if}\ y \geq y^\ast,$$ where $l(x)$ is fit only to the good configurations (loss below $y^\ast$) and $g(x)$ only to the bad ones. There is no regression of $y$ on $x$ at all — just "what do good configs look like" versus "what do bad configs look like." The choice of $y^\ast$ is forced, not free: the GP instinct is to set $y^\ast$ to the single best loss seen, but then "good" is one configuration and I cannot estimate a density from one point. So $y^\ast$ is a *quantile* of the observed losses, $\gamma = p(y < y^\ast)$, the best ~15–25%, chosen precisely so $l$ is estimable from more than a handful of points; and I notice $\gamma$ is the only number of $p(y)$ I will ever need.

The payoff is that EI collapses to a density ratio — and I do not have to assume this, the integral forces it. Writing $p(y \mid x) = p(x \mid y)\, p(y) / p(x)$ and substituting, the denominator factors out because $p(x) = \int p(x \mid y)\, p(y)\, dy = \gamma\, l(x) + (1-\gamma)\, g(x)$, splitting the total-probability integral at $y^\ast$ and using $\int_{y < y^\ast} p(y)\, dy = \gamma$. In the numerator the integration runs over $y < y^\ast$, where $p(x \mid y) \equiv l(x)$ pulls straight out: $\int_{-\infty}^{y^\ast} (y^\ast - y)\, p(x \mid y)\, p(y)\, dy = l(x)\,\big[\gamma y^\ast - \int_{-\infty}^{y^\ast} y\, p(y)\, dy\big]$. Calling the bracket $A = \gamma y^\ast - \int_{-\infty}^{y^\ast} y\, p(y)\, dy$ — a positive constant with no dependence on $x$ — and dividing numerator and denominator through by $l(x)$ gives $$\mathrm{EI}_{y^\ast}(x) = A \cdot \Big( \gamma + (1-\gamma)\, \frac{g(x)}{l(x)} \Big)^{-1} \ \propto\ \Big( \gamma + \frac{g(x)}{l(x)}(1-\gamma) \Big)^{-1}.$$ Here $x$ enters only through the ratio $g(x)/l(x)$, and EI is decreasing in it, so maximizing EI is exactly minimizing $g(x)/l(x)$, i.e. maximizing $l(x)/g(x)$: propose the configuration most probable under the good density and least probable under the bad one. I never evaluate $A$, never model $p(y)$ beyond the single number $\gamma$, never compute $\int y\, p(y)\, dy$ — the argmax over $x$ does not see them.

This is where modeling $p(x \mid y)$ pays off twice. Maximizing $l(x)/g(x)$ needs no global optimizer over an opaque surface, because $l(x)$ is a density I can *sample from*: draw a batch of candidate configurations from $l$ — already concentrated where the good configs live — score each by $l(x)/g(x)$, and keep the best. No CMA-ES, no EDA, no restarts; the proposed direction and the modeled direction are now aligned. And I build $l$ and $g$ by taking the generative prior I already have and re-estimating each of its leaf distributions from the relevant subset of observations — keeping the conditional graph exactly as it was (first layers, then per-layer parameters) but replacing each node's prior distribution with a non-parametric density fit to the good observations that reached that node (for $l$) or the bad ones (for $g$). Because it is the same generative graph, sampling from $l$ respects the conditional structure automatically and the cost is *linear* in history size and dimension — no cubic GP, no hand-grouping. The per-node estimator is an *adaptive Parzen* (kernel) density. For a continuous variable the prior draws uniform on a box $(a, b)$, I estimate the density as an equally-weighted mixture of one Gaussian per observed value plus the original prior kept in the mixture as a broad Gaussian at the box midpoint with width of order $(b-a)$, so the density never collapses to zero away from data and a node seen by one or two good trials still has sensible spread. The one real choice is each Gaussian's bandwidth: a single global bandwidth would over-smooth in dense regions and under-smooth in sparse ones, so I make it adaptive per point — sort the values and set each point's standard deviation to the *greater* of the distances to its sorted left and right neighbor, with the box endpoints $a$ and $b$ counting as neighbors at the ends. Where good points cluster tightly the gaps are small, kernels are narrow, and $l$ is sharply peaked exactly where I am confident; where points are sparse the gaps are large, kernels wide, and $l$ stays diffuse where I am ignorant. To keep it numerically sane the bandwidth is clipped to $[\text{prior}\_\sigma / \min(100, 1+N),\ \text{prior}\_\sigma]$ — a floor so kernels never become spikes, a ceiling at the prior width. The other variable types fall out by analogy because the prior names them: a log-uniform prior (a learning rate spanning decades) gets the identical mixture-of-neighbors construction in the log domain, and a categorical prior with probabilities $p_i$ is re-weighted so the posterior probability of choice $i$ is proportional to $N p_i + C_i$, where $C_i$ counts occurrences of choice $i$ among the good observations and $N$ is the number of choices — the prior counts $N p_i$ acting as pseudo-counts that keep every category alive when data are scarce, the observed counts tilting toward what good trials favored (a Parzen window with a categorical kernel). Two practical knobs round it out. As the history grows the search drifts and old trials reflect a region I have moved past, so I optionally weight observations by recency — linear forgetting — letting the most recent count fully and ramping down the oldest; it is a robustness knob, not part of the EI derivation. And the quantile is in practice gentler than a flat fraction: let the number of good points grow like $\sqrt{N}$ (capped) so $l$ is neither starved early nor bloated late. For scoring I compare candidates by $\log l(x) - \log g(x)$ rather than the raw ratio, since the densities are products over conditional nodes and underflow fast. The assembled loop is a random warm-up of a few dozen trials (the split is meaningless with almost no history), then each iteration: sort the history by loss, split at the $\gamma$ quantile into good and bad; build $l$ from the good and $g$ from the bad by adaptive Parzen / re-weighted categoricals per node; sample a couple dozen candidates from $l$; score each by $\log l - \log g$; evaluate the true objective at the best — the single expensive call — append, repeat.

```python
import numpy as np


def adaptive_parzen_normal(obs, a, b, prior_weight=1.0):
    """Equally-weighted Gaussian mixture over (a, b): one kernel per observation
    plus a broad prior Gaussian at the box midpoint. Per-point bandwidth = the
    greater distance to a sorted neighbor (box endpoints count as neighbors),
    clipped to [prior_sigma/min(100,1+N), prior_sigma]."""
    obs = np.asarray(obs, float)
    prior_mu, prior_sigma = 0.5 * (a + b), (b - a)
    mus = np.concatenate([obs, [prior_mu]])              # data kernels + prior
    order = np.argsort(mus)
    smus = mus[order]
    sigma = np.empty_like(smus)
    if len(smus) > 2:
        sigma[1:-1] = np.maximum(smus[1:-1] - smus[:-2], smus[2:] - smus[1:-1])
        sigma[0]  = smus[1]  - smus[0]
        sigma[-1] = smus[-1] - smus[-2]
    else:
        sigma[:] = prior_sigma
    sig = np.empty_like(sigma)
    sig[order] = sigma                                   # unsort to obs order
    minsig = prior_sigma / min(100.0, 1.0 + len(mus))    # floor: no spikes
    sig = np.clip(sig, minsig, prior_sigma)              # ceiling: prior width
    w = np.ones(len(mus)); w[-1] = prior_weight          # prior kept in the mix
    return w / w.sum(), mus, sig


def normal_lpdf(x, w, mus, sigmas):
    comp = -0.5 * ((x - mus) / sigmas) ** 2 - np.log(sigmas * np.sqrt(2 * np.pi))
    return np.log(np.sum(w * np.exp(comp)) + 1e-300)


def sample_mixture(w, mus, sigmas, a, b, rng):
    while True:                                          # truncate to the box
        k = rng.choice(len(w), p=w)
        x = rng.normal(mus[k], sigmas[k])
        if a <= x <= b:
            return x


def tpe(objective, a, b, n_init=20, max_trials=120,
        gamma=0.25, n_candidates=24, seed=0):
    rng = np.random.default_rng(seed)
    X = [rng.uniform(a, b) for _ in range(n_init)]       # random warm-up
    y = [objective(x) for x in X]                        # expensive calls

    for _ in range(max_trials - n_init):
        order = np.argsort(y)                            # split history by loss
        n_below = max(1, int(np.ceil(gamma * np.sqrt(len(y)))))
        below = [X[i] for i in order[:n_below]]          # good -> l(x)
        above = [X[i] for i in order[n_below:]]          # bad  -> g(x)

        wl, ml, sl = adaptive_parzen_normal(below, a, b)
        wg, mg, sg = adaptive_parzen_normal(above, a, b)

        best_x, best_score = None, -np.inf               # sample from l, rank by l/g
        for _ in range(n_candidates):
            xc = sample_mixture(wl, ml, sl, a, b, rng)
            score = normal_lpdf(xc, wl, ml, sl) - normal_lpdf(xc, wg, mg, sg)
            if score > best_score:                       # argmax (log l - log g) = argmax EI
                best_x, best_score = xc, score

        X.append(best_x); y.append(objective(best_x))    # one expensive trial
    i = int(np.argmin(y))
    return X[i], y[i]


if __name__ == "__main__":
    # toy expensive objective: a multimodal 1-D loss on [-10, 10]
    f = lambda x: np.sin(x) + 0.1 * (x - 2.0) ** 2
    x_best, y_best = tpe(f, -10.0, 10.0, seed=0)
    print(f"best x = {x_best:.4f}, loss = {y_best:.4f}")
```

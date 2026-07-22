OK, let me start from where it actually hurts. I'm tuning a deep belief network — really any of these deep models — and I have somewhere between ten and fifty hyper-parameters: a classifier learning rate, an annealing start, an L2 penalty, a pre-processing choice, a number of layers, and then *per layer* a hidden-unit count, a weight-init scheme, contrastive-divergence epochs and learning rate and annealing. Each "trial" is one full configuration turned into a trained, evaluated model, and that costs real wall-clock. So I get tens to maybe a couple hundred trials, total. That's the budget.

Two more things make this not a vanilla box-optimization problem. The variables are of mixed type — continuous (a learning rate), ordinal (hidden units), categorical (raw vs ZCA pre-processing). And the space is *conditional*: "number of units in the second layer" is only a meaningful variable when "number of layers" came out ≥ 2. So the configuration isn't a fixed-length vector; the very set of active variables depends on choices made higher up. An optimizer here has to decide *which* variables to optimize at the same time as it decides their values.

Let me work through the obvious things first, because where each one runs out of room is what tells me what I actually need.

Grid search? Exponential in the number of hyper-parameters, and it spends resolution on axes the loss barely cares about. Dead on arrival at fifty dimensions. Manual tuning is what people actually do, and humans are genuinely good at it when only a handful of trials are affordable — but it doesn't scale, it's unreproducible, and it can't systematically exploit a cluster.

Random search is more interesting. The clean way to even *describe* this conditional, mixed-type space is to write it as a generative process: draw the number of layers; conditioned on that, draw each existing layer's parameters; draw the pre-processing; and so on. A valid configuration is exactly a draw of that process, and "which variables are active" falls out of the draw for free — no special-casing. So random search is just: sample from that generative prior, evaluate, repeat. And it's known to beat grid search (Bergstra & Bengio 2012) precisely because grid search wastes trials enumerating insensitive coordinates while random spreads its trials across all of them; for one-layer nets a few dozen random trials match careful manual tuning. So random search is my baseline, and it has the killer property that it handles the tree structure trivially — the prior already encodes it.

But here's the crack: random search is *memoryless*. It draws from the same prior every time and never once looks at the losses it has already measured. On the hard DBN datasets — convex, the rotated-MNIST one — it converges slowly or plateaus below what careful search reaches. Every trial is gold and I'm throwing the information away. What I want is to *use the history* to steer the next draw, while keeping random search's effortless handling of the conditional space.

So the frame is sequential model-based optimization. Replace the expensive `f(x)` with a cheap surrogate `M` fit to the history `H = {(x_i, f(x_i))}`; at each step optimize a cheap criterion `S(x, M)` to pick the next `x`; evaluate the true `f` there (the one expensive call); append; refit. The loop is generic — `argmin_x S(x, M)`, evaluate, append, refit — and everything rides on two choices: what surrogate models `f`, and what criterion `S` I optimize.

The criterion first, because it's the more settled question. Expected Improvement. Pick a threshold `y*`; given a model `M` that gives me a distribution over the value `y` at a candidate `x`, score `x` by how much improvement below `y*` I expect:

  EI_{y*}(x) = ∫_{−∞}^{y*} (y* − y) p_M(y | x) dy.

This is the right object — it weights a possible gain by both how big it is and how likely, so it trades exploitation against exploration on its own without a knob I have to hand-tune per problem. EI is non-negative, zero where I've already evaluated, multimodal — iterate it and you search globally. Good. EI stays.

Now the surrogate, and this is where I have to actually think.

The textbook answer is a Gaussian process over `p(y | x)`: a prior over functions, closed under conditioning, so the posterior given `H` is again a GP with closed-form mean and variance, and EI gets a closed form from that mean and variance. Lovely on a clean box. The question is what happens when I try to fit it to *my* problem, so let me work through what each step of the GP recipe actually demands here.

First cost: fitting the GP is `O(|H|^3)` in the number of trials — tolerable while the expensive evaluations dominate, but it's there.

Second, the one that bites harder: EI on a GP is a multimodal surface I now have to *globally optimize* over a ten-to-fifty-dimensional mixed-type space. There's no gradient I trust, the discrete and continuous parts need different machinery — an estimation-of-distribution scheme on the categoricals, a CMA-ES on the continuous part, restarts from promising simplex centers. That's a heavy auxiliary optimizer wrapped around an opaque criterion, and it's only optimizing a *cheap* function, yet it's most of the engineering.

Third, and this is the one I keep snagging on: the tree structure. A single GP wants a fixed-dimensional input vector with a kernel measuring distance between two configurations. But my configurations don't have a fixed set of coordinates — a one-layer config and a three-layer config don't even live in the same space. The retrofit is to *group* the hyper-parameters and place an independent GP over each group — one over the common parameters, one per layer, little one-dimensional GPs over stray conditionals. It works, but it's carving the conditional structure up by hand and gluing GPs together. The thing random search did for free — respect the generative tree — the GP fights me on.

Let me sit with that. The GP models `p(y | x)`: given a configuration, what's the distribution of losses? To *propose* a configuration I then have to invert that — search `x`-space for where the predicted loss is promising. The expensive direction (searching configs) and the modeled direction (predicting loss from a config) are opposed, and the opposition is exactly what forces the heavy auxiliary optimizer and what makes the tree structure awkward, because the model is indexed by the full config vector.

What if I model the *other* direction? Instead of `p(y | x)`, model `p(x | y)` — given a level of loss, what do configurations look like? And `p(y)` on the side. At first that feels backwards; `p(x | y)` isn't obviously a thing I can plug into EI. But notice what `p(x | y)` *is*: it's a distribution over *configurations*. And I already have, from random search, an entire apparatus for representing and sampling distributions over configurations — the generative process. If I can express `p(x | y)` as a generative process of the same shape as the prior, then sampling configurations is free and the tree structure is respected automatically, exactly as it was for random search. That's the property the GP kept costing me. Let me chase this.

How would `p(x | y)` even be defined from data? I have a pile of `(x_i, y_i)`. The cleanest thing I can do with a pile of losses is *threshold* them. Pick a threshold `y*` and split: configurations whose loss came out below `y*` are the "good" ones, the rest are "bad". So model the conditional as two densities:

  p(x | y) = l(x) if y < y* ;  g(x) if y ≥ y*,

where `l(x)` is the density over the good configs and `g(x)` over the bad ones. Two densities over configuration space, each estimated only from its own subset of the history. That's it — that's the model. No regression of `y` on `x` at all; just "what do good configs look like" versus "what do bad configs look like."

Now, what should `y*` be? The instinct from the GP world is to set `y*` to the *best loss seen so far* — maximally aggressive, EI measures improvement over the incumbent. But watch what that does here: if `y*` is the single best value, then "good" is *one* configuration. I can't estimate a density `l(x)` from one point. So aggressive-`y*` is fine when the model is a GP that doesn't care how many points are below the line, but it's poison for a density-splitting model. I need enough points below `y*` to fit `l(x)`. So set `y*` to a *quantile* of the observed losses: `γ = p(y < y*)`, some fixed fraction — the best 15–25% of trials go into `l`, the rest into `g`. The quantile is forced on me by the need to estimate `l(x)` from more than a handful of points; it's not an arbitrary hyper-parameter, it's what makes the density estimable. And I notice I don't actually need to model `p(y)` in any detail — all I need from it is this one number `γ`.

But now I owe myself the bridge: I've defined `p(x | y)`, and EI is written in terms of `p(y | x)`. Does scoring by these two densities actually correspond to maximizing expected improvement? If it doesn't, I've built a pretty model that optimizes the wrong thing. Let me grind the integral out and see.

EI wants `p(y | x)`; I have `p(x | y)` and `p(y)`. Bayes: `p(y | x) = p(x | y) p(y) / p(x)`. Substitute into EI:

  EI_{y*}(x) = ∫_{−∞}^{y*} (y* − y) p(y | x) dy = ∫_{−∞}^{y*} (y* − y) · p(x | y) p(y) / p(x) dy.

Take the denominator `p(x)` first, since it doesn't depend on `y` and pulls out. By the law of total probability,

  p(x) = ∫_ℝ p(x | y) p(y) dy.

Split that integral at `y*`. For `y < y*`, `p(x | y) = l(x)`, and `∫_{y<y*} p(y) dy = γ` by the definition of the quantile. For `y ≥ y*`, `p(x | y) = g(x)`, and `∫_{y≥y*} p(y) dy = 1 − γ`. So

  p(x) = γ l(x) + (1 − γ) g(x).

Clean. Now the numerator integral. On the whole range of integration `y` runs from `−∞` to `y*`, i.e. `y < y*`, so `p(x | y)` is identically `l(x)` there and pulls straight out of the integral:

  ∫_{−∞}^{y*} (y* − y) p(x | y) p(y) dy = l(x) ∫_{−∞}^{y*} (y* − y) p(y) dy.

Expand the remaining integral:

  l(x) [ y* ∫_{−∞}^{y*} p(y) dy − ∫_{−∞}^{y*} y p(y) dy ] = l(x) [ γ y* − ∫_{−∞}^{y*} y p(y) dy ],

using `∫_{−∞}^{y*} p(y) dy = γ` again. So putting numerator over denominator,

  EI_{y*}(x) = [ γ y* l(x) − l(x) ∫_{−∞}^{y*} y p(y) dy ] / [ γ l(x) + (1 − γ) g(x) ].

Now stare at the structure. The whole numerator has a factor of `l(x)`; pull it out, and call the bracket `A = γ y* − ∫_{−∞}^{y*} y p(y) dy`. That `A` has *no dependence on `x`* — it's built only from `y*`, `γ`, and the marginal `p(y)`. It's a positive constant (on the region `y < y*` the integrand `y p(y)` integrates to something below `γ y*`, since every `y` there is below `y*`). Divide numerator and denominator through by `l(x)`:

  EI_{y*}(x) = A · ( γ + (1 − γ) g(x)/l(x) )^{−1} ∝ ( γ + g(x)/l(x) (1 − γ) )^{−1}.

There it is. EI is a *constant times the inverse of* `γ + (1−γ) g(x)/l(x)`. The only place `x` enters is through the ratio `g(x)/l(x)`. And EI is *decreasing* in that ratio — bigger `g/l` makes the bracket bigger makes EI smaller. So:

  maximize EI  ⇔  minimize g(x)/l(x)  ⇔  **maximize l(x)/g(x).**

So expected improvement, the principled acquisition I started from, comes out proportional to `(γ + (1−γ) g(x)/l(x))^{−1}` — which is monotone decreasing in `g/l`, so maximizing it is the same as maximizing `l(x)/g(x)`. Said in words: find the configuration that is most probable under the good-density `l` and least probable under the bad-density `g`. That matches the intuition — I want configs that look like the ones that worked and not like the ones that didn't — but I didn't assume the intuition, I pushed the EI integral through Bayes and this is what fell out.

I want to be careful here, though, because that derivation had two spots I could have fooled myself. I split `p(x)` at `y*` and called the tail masses `γ` and `1−γ`; I pulled `l(x)` out of the numerator integral on the strength of "`p(x|y) = l(x)` for all `y < y*`"; and I asserted `A > 0` from a one-line argument. A sign slip or a botched normalization anywhere there would still *look* like a clean formula. So before I build a whole method on it, let me check the closed form against the EI integral evaluated numerically on a concrete instance — if they don't agree pointwise, I've made an error in the algebra and I need to find it.

Take `p(y) = N(0,1)` for the marginal over losses, `γ = 0.25` so `y* = Φ^{−1}(0.25) = −0.6745`, and pick two arbitrary valid densities for the conditional pieces: `l = N(1, 0.7)` (good configs near `x=1`), `g = N(−0.5, 1.5)` (bad ones spread near `x=−0.5`). Build `p(x|y) = l` for `y<y*` else `g`, set `p(y|x) = p(x|y)p(y)/p(x)`, and quadrature the *definition* `∫_{−∞}^{y*}(y*−y)p(y|x)dy` directly. Against that, evaluate my closed form `A·(γ + (1−γ)g/l)^{−1}` with `A = γy* − ∫_{−∞}^{y*} y p(y) dy`. The constant comes out `A = 0.1492` — positive, as the structural argument promised. Then pointwise:

```
   x      EI_numeric      EI_closed         g/l
-2.00     0.00007215     0.00007215     2755.99
-1.00     0.00750868     0.00750868       26.15
 0.00     0.12764141     0.12764141        1.22
 0.50     0.24383575     0.24383575        0.48
 1.00     0.32264490     0.32264490        0.28
 1.50     0.34233017     0.34233017        0.25
 2.00     0.30308206     0.30308206        0.32
 3.00     0.09247782     0.09247782        1.82
```

Identical to eight decimals at every `x`. So the algebra is right — no dropped sign, no normalization I forgot — and the EI surface really does fall as `g/l` rises (the EI column peaks at `x=1.5`, exactly where `g/l` bottoms out at 0.25). Sweeping a fine grid, `argmax EI` and `argmax l/g` land on the same point, `x = 1.4171`. That's the thing I needed to trust: I never have to evaluate `A`, never have to model `p(y)` beyond the single number `γ`, never compute `∫ y p(y) dy` — they all sit in the `x`-independent constant the argmax can't see. EI collapses to a density ratio, and now I've watched it do so on actual numbers rather than just trusting the manipulation.

And this is where modeling `p(x|y)` pays off twice over. Maximizing `l(x)/g(x)` doesn't need a global optimizer over an opaque surface. `l(x)` is a density I can *sample from*. So: draw a batch of candidate configurations from `l` — they're already concentrated where the good configs live — score each by `l(x)/g(x)`, and keep the best. No CMA-ES, no EDA, no restarts. Just sample-from-good, rank-by-ratio. The expensive direction (proposing configs) and the modeled direction are now *aligned* — I model configurations and I propose by sampling configurations.

Now I have to actually build `l(x)` and `g(x)` as densities over the configuration space — and do it so the tree structure survives. The move: take the generative prior I already have, and *re-estimate each of its leaf distributions from the relevant subset of observations*. Keep the conditional graph exactly as it was — first choose layers, then per-layer parameters — but replace each prior distribution at a node with a non-parametric density fit to the good observations that reached that node (for `l`) or the bad ones (for `g`). Because it's the same generative graph, sampling from `l` respects the conditional structure automatically, and the runtime is *linear* in the history size and in the number of variables — no cubic GP fit, no carving the space into groups by hand. The tree structure that the GP fought is now free, just like in random search.

What non-parametric density per node? Parzen windows. Take a continuous variable that the prior draws uniform on `(a, b)`. Given the good observations `{x^(1), …, x^(k)}` of that variable, estimate its density as an equally-weighted mixture: one Gaussian centered at each observed value, plus the original prior kept in the mixture so the density never collapses to zero away from data (and so a node seen by only one or two good trials still has sensible spread). That prior term is a broad Gaussian sitting at the middle of the box with a width of order `(b − a)`, given its own weight in the mixture.

The one real choice is each Gaussian's bandwidth. If I use a single global bandwidth I'll over-smooth in dense regions and under-smooth in sparse ones. So make it *adaptive per point*: sort the observed values; set each point's standard deviation to the *greater* of the distances to its left and right neighbor, treating the box endpoints `a` and `b` as neighbors for the end points. Where good points cluster tightly, the neighbor gaps are small, so the kernels are narrow and `l` is sharply peaked there — exactly where I'm confident; where points are sparse, the gaps are large, the kernels wide, and `l` stays diffuse — where I'm ignorant. To keep it numerically sane I clip the bandwidth to a range — a floor of roughly the prior width divided by the number of points so kernels never become spikes, and a ceiling at about the prior width. This "adaptive Parzen" is the whole density estimator.

The other variable types fall out by analogy, because the prior already names them. A *log-uniform* prior (a learning rate spanning decades) — do the identical Gaussian-mixture-of-neighbors construction but in the log domain, which is an exponentiated truncated Gaussian mixture back in the original units; the bandwidths adapt in log-space, matching how I actually think about a learning rate. A *categorical* prior with probabilities `p_i` — there's nothing continuous to place a Gaussian on, so re-weight the categories by how often each good config used them: the posterior probability of choice `i` is proportional to `N p_i + C_i`, where `C_i` counts occurrences of choice `i` among the good observations and `N` is the number of choices. The prior counts `N p_i` act as pseudo-counts that keep every category alive when the data are scarce, and the observed counts `C_i` tilt toward the choices the good trials favored. That's just a Parzen window with a categorical kernel.

A subtlety once the history grows: the search drifts, and old trials reflect a region I've moved away from. So weight the observations when building `l` and `g`, letting the most recent ones count fully and ramping down the weight of the oldest — linear forgetting. It's a robustness knob, not part of the EI derivation, but it keeps the densities tracking where the search currently is rather than averaging in stale early trials. I'll keep it modest.

Let me also pin the constants. The quantile `γ`: small enough that `l` captures genuinely good configs, large enough to estimate a density from — a quarter or so works; the "good" set is the best ~15–25%. (In practice it's even gentler than a flat quantile — let the number of good points grow like `√(number of trials)` so `l` doesn't get starved early or bloated late, but the idea is the top fraction.) The number of candidates drawn from `l` per step: a couple dozen — enough that the best-ratio candidate is a good EI maximizer, cheap enough to score instantly. And a random warm-up of a few dozen trials before TPE engages, because with almost no history the split into good and bad is meaningless. One numerical note for scoring: compare candidates by `log l(x) − log g(x)` rather than the raw ratio — the densities are products over conditional nodes and underflow fast, so work in log space and pick the maximum.

Let me assemble the loop. Random warm-up draws from the prior. Then each iteration: sort the history by loss, split at the `γ` quantile into good and bad; build `l` from the good (adaptive Parzen / re-weighted categoricals per node), build `g` from the bad; sample a couple dozen candidate configs from `l`; score each by `log l − log g`; evaluate the true objective at the best one — the single expensive call — append, repeat.

```python
import numpy as np

# --- adaptive Parzen estimator for one continuous variable on a box (a, b) -----
def adaptive_parzen_normal(obs, a, b, prior_weight=1.0):
    """Equally-weighted Gaussian mixture: one kernel per observation + a broad
    prior Gaussian at the box midpoint. Per-point bandwidth = greater distance
    to a sorted neighbor (endpoints count as neighbors), clipped to a range."""
    obs = np.asarray(obs, float)
    prior_mu, prior_sigma = 0.5 * (a + b), (b - a)
    mus = np.concatenate([obs, [prior_mu]])             # data kernels + prior
    order = np.argsort(mus)
    smus = mus[order]
    sigma = np.empty_like(smus)
    if len(smus) > 2:                                   # interior: max neighbor gap
        sigma[1:-1] = np.maximum(smus[1:-1] - smus[:-2], smus[2:] - smus[1:-1])
        sigma[0]  = smus[1]  - smus[0]
        sigma[-1] = smus[-1] - smus[-2]
    else:
        sigma[:] = prior_sigma
    sig = np.empty_like(sigma); sig[order] = sigma      # unsort
    minsig = prior_sigma / min(100.0, 1.0 + len(mus))   # floor: no spikes
    sig = np.clip(sig, minsig, prior_sigma)             # ceiling: prior width
    w = np.ones(len(mus)); w[-1] = prior_weight         # prior kept in the mix
    return w / w.sum(), mus, sig

def normal_lpdf(x, w, mus, sigmas):
    """log-density of a Gaussian mixture, evaluated at scalar x."""
    comp = -0.5 * ((x - mus) / sigmas) ** 2 - np.log(sigmas * np.sqrt(2 * np.pi))
    return np.log(np.sum(w * np.exp(comp)) + 1e-300)

def sample_mixture(w, mus, sigmas, a, b, rng):
    while True:                                          # truncate to the box
        k = rng.choice(len(w), p=w)
        x = rng.normal(mus[k], sigmas[k])
        if a <= x <= b:
            return x

# --- TPE for a 1-D objective (the SMBO outer loop) ----------------------------
def tpe(objective, a, b, n_init=20, max_trials=120,
        gamma=0.25, n_candidates=24, seed=0):
    rng = np.random.default_rng(seed)
    X = [rng.uniform(a, b) for _ in range(n_init)]      # random warm-up
    y = [objective(x) for x in X]                        # expensive calls

    for _ in range(max_trials - n_init):
        order = np.argsort(y)                            # split history by loss
        n_below = max(1, int(np.ceil(gamma * np.sqrt(len(y)))))   # top ~gamma
        below = [X[i] for i in order[:n_below]]          # good  -> l(x)
        above = [X[i] for i in order[n_below:]]          # bad   -> g(x)

        wl, ml, sl = adaptive_parzen_normal(below, a, b) # l(x)
        wg, mg, sg = adaptive_parzen_normal(above, a, b) # g(x)

        # draw candidates from l, keep the one that maximizes log l - log g
        best_x, best_score = None, -np.inf
        for _ in range(n_candidates):
            xc = sample_mixture(wl, ml, sl, a, b, rng)   # sample from the good density
            score = normal_lpdf(xc, wl, ml, sl) - normal_lpdf(xc, wg, mg, sg)
            if score > best_score:                       # argmax l/g  ==  argmax EI
                best_x, best_score = xc, score

        X.append(best_x); y.append(objective(best_x))    # one expensive trial
    i = int(np.argmin(y))
    return X[i], y[i]
```

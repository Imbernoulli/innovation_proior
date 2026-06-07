# Tree-structured Parzen Estimator (TPE): density-ratio SMBO for hyper-parameter optimization

## Problem

Optimize a loss `f(x)` over a **tree-structured (conditional), mixed-type configuration space** —
ten to fifty hyper-parameters where some variables (2nd-layer size) only exist when a parent
variable (number of layers) takes the right value — using **few, expensive trials** (each trial
trains and evaluates a model). The optimizer must use its history (unlike random search), respect
the conditional structure (which a single Gaussian process resists), and avoid globally optimizing
an opaque acquisition over a high-dimensional mixed space.

## Key idea

Stay in the SMBO frame (fit a cheap surrogate to the history, optimize Expected Improvement to
pick the next trial), but **model the generative direction `p(x|y)` instead of `p(y|x)`**. Split
the history at a quantile threshold `y*` and represent the conditional as two densities over
configurations:

`p(x|y) = l(x)` if `y < y*`, `g(x)` if `y ≥ y*`,

where `l(x)` is fit to the **good** configurations (loss below `y*`) and `g(x)` to the **bad**
ones. The threshold is a **quantile**, `γ = p(y < y*)` (best ~15–25%), chosen so `l` is estimated
from more than one point — an aggressive `y* = best` would leave a single good point. Only this one
number `γ` of `p(y)` is ever needed.

Expected Improvement then collapses to a density ratio. With `EI_{y*}(x) = ∫_{−∞}^{y*}(y*−y)p(y|x)dy`
and `p(y|x) = p(x|y)p(y)/p(x)`:

- denominator `p(x) = ∫ p(x|y)p(y)dy = γ·l(x) + (1−γ)·g(x)`;
- numerator `∫_{−∞}^{y*}(y*−y)p(x|y)p(y)dy = l(x)·[γ y* − ∫_{−∞}^{y*} y·p(y)dy]` (since `p(x|y)=l(x)`
  on `y < y*` and `∫_{−∞}^{y*}p(y)dy = γ`).

Let `A = γ y* − ∫_{−∞}^{y*} y·p(y)dy` (a positive constant, independent of `x`). Dividing through
by `l(x)`:

**`EI_{y*}(x) = A · ( γ + (1−γ)·g(x)/l(x) )^{−1} ∝ ( γ + (g(x)/l(x))(1−γ) )^{−1}`.**

`x` enters only through `g(x)/l(x)`, and EI decreases in it, so

**maximize EI ⇔ minimize `g(x)/l(x)` ⇔ maximize `l(x)/g(x)`** — propose the configuration most
probable under the good density and least probable under the bad one. Because `l(x)` is a density
we can *sample*, the acquisition needs no auxiliary optimizer: draw candidates from `l`, score by
`l(x)/g(x)` (in practice `log l − log g`), keep the best.

`l` and `g` are the **generative prior with each leaf distribution re-estimated by an adaptive
Parzen estimator** from the good / bad subset — same conditional graph, so the tree structure is
respected for free and the cost is **linear** in history size and in dimension (no `O(|H|^3)` GP,
no hand-grouping into separate GPs).

## Algorithm

1. **Warm-up.** Run `n_startup` (~20) random trials drawn from the prior generative process.
2. **Split.** Sort the history by loss; the best `γ`-quantile forms the good set (→ `l`), the rest
   the bad set (→ `g`). (Hyperopt uses `n_below = min(⌈γ·√N⌉, 25)` good points; `γ = 0.25`.)
3. **Fit densities.** Per variable, build an **adaptive Gaussian Parzen mixture**: one Gaussian per
   observation plus the box prior kept in the mixture; per-point bandwidth = the **greater of the
   distances to the sorted left/right neighbor** (endpoints count as neighbors), clipped to
   `[prior_σ/min(100, 1+N), prior_σ]`. Log-uniform variables: same construction in the log domain.
   Categorical variables: re-weighted categorical, posterior ∝ `N·p_i + C_i` (`C_i` = count of
   choice `i` among good observations). Optionally linear-forget old observations.
4. **Acquire.** Sample `n_EI_candidates` (~24) configurations from `l`; score each by
   `log l(x) − log g(x)`; take the maximizer.
5. **Evaluate** the true objective there (the one expensive call), append to the history, repeat
   from step 2.

## Code

A faithful, self-contained 1-D realization showing the adaptive-Parzen `l`/`g`, sampling from `l`,
the `l/g` scoring, and the SMBO outer loop (mirrors the structure of `hyperopt`'s
`adaptive_parzen_normal`, the `γ`-split, and the `below_llik − above_llik` candidate scoring).

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

For real conditional, mixed-type spaces the same logic runs over the generative graph: each
node's leaf distribution (uniform → truncated Gaussian mixture, log-uniform → exponentiated
truncated Gaussian mixture, categorical → re-weighted categorical) is re-fit from the good / bad
subsets, candidates are sampled by walking the generative process under `l`, and scored by
`log l − log g` summed over the active nodes — exactly the design realized in `hyperopt`'s
`tpe.suggest` (`γ = 0.25`, `n_below = min(⌈γ√N⌉, 25)`, `n_EI_candidates = 24`,
`n_startup_jobs = 20`, linear-forgetting `LF = 25`).

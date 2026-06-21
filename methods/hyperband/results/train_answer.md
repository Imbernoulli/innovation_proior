The problem is black-box hyperparameter optimization under a fixed resource budget. We have a configuration space and an expensive train-and-validate operation. We want to find a configuration that minimizes validation error without exceeding a total compute budget. The objective is unknown, non-convex, high-dimensional, and noisy. We only observe it through partial training runs, and two critical quantities are hidden from us: how fast each configuration's validation curve converges, and how rare good configurations are in the search space.

Existing methods attack only one side of this. Random search is a strong baseline because it spreads trials across the axes that matter and needs no smoothness assumptions, but it trains every configuration to the maximum resource R. That caps the number of configurations we can examine at B/R and wastes full runs on hopeless configurations that reveal themselves after a tiny fraction of R. Bayesian optimization methods such as TPE, SMAC, and Spearmint are smarter about which configuration to try next, but they still train each selected configuration to completion. They solve adaptive selection, not adaptive evaluation, so they inherit the same per-evaluation cost. Early-stopping hybrids that try to learn convergence curves reintroduce parametric assumptions that are hard to verify and break badly when wrong.

The better lever is to adapt how much resource we spend on each configuration before deciding it is uncompetitive. Treat each configuration as an arm whose partial loss after k units of resource converges to an unknown terminal value. Two configurations become separable once the envelope that bounds partial losses shrinks below half their terminal gap, so each configuration has its own separation cost. Uniform allocation pays the hardest configuration's separation cost on every configuration. Successive Halving fixes this by running elimination rounds: in each round every surviving configuration receives the same resource, the worst fraction is dropped, and the resource escalates geometrically onto survivors. Bad configurations are cut after small resource; competitive ones receive more.

But Successive Halving requires the initial number of configurations n as an input, and the right n depends on the very unknowns we cannot see. If we pick n too large, each configuration gets too little resource and a good one may be eliminated because its partial loss is still noisy. If we pick n too small, we may never sample a good configuration at all. We cannot reliably estimate the convergence envelope or the distribution of terminal values without reintroducing fragile assumptions.

Hyperband solves this by not choosing n at all. It hedges over the entire n-versus-B/n tradeoff by running Successive Halving for a geometric grid of bracket sizes. Each bracket is indexed by s. The most aggressive bracket starts with many configurations at very low resource; the least aggressive bracket is plain random search at full resource. With only O(log R) brackets, the hedge costs at most a logarithmic factor over knowing the best n in advance.

Concretely, fix a maximum resource R per configuration and a halving rate eta, typically 3. Set s_max to floor(log_eta R) and a per-bracket budget B to (s_max + 1) R. For each s from s_max down to 0, the starting pool size is n = ceil((B/R) * eta^s / (s+1)) and the minimum resource is r = R * eta^{-s}. This guarantees that after s promotions, each multiplying resource by eta, the survivor has reached R. The inner Successive Halving loop runs rounds i = 0 to s, keeping the top floor(n_i/eta) configurations and multiplying resource by eta each round. After all brackets finish, return the configuration with the smallest validation loss observed anywhere.

The eta parameter controls how aggressively each bracket cuts: larger eta means fewer, sharper cuts; smaller eta means more, gentler cuts. The bracket count is logarithmic in R, so trying all tradeoffs is cheap. Uniform i.i.d. sampling is the default because the analysis needs only independent draws from a stationary distribution; a smarter sampler can be swapped in but is optional. The result is provably close to Successive Halving with the optimal n up to log factors, and matches known best-arm lower bounds in stochastic settings.

```python
import numpy as np
from math import log, ceil


def successive_halving(space, n, r, s, eta, rng,
                       sample_configuration,
                       run_then_return_val_loss, top_k):
    """One bracket: start with n configs at minimum resource r, and over s+1 rounds
    keep the top 1/eta while multiplying the resource by eta each round."""
    T = [sample_configuration(space, rng) for _ in range(n)]
    seen = []
    for i in range(s + 1):
        n_i = int(n * eta ** (-i))
        r_i = r * eta ** i
        losses = [run_then_return_val_loss(t, r_i) for t in T]
        seen.extend(zip(T, losses, [r_i] * len(T)))
        T = top_k(T, losses, int(n_i / eta))
    return seen


def hyperband(space, R, eta, rng,
              sample_configuration,
              run_then_return_val_loss, top_k):
    """Hedge Successive Halving over bracket sizes from maximal early stopping
    to plain random search. Each bracket uses roughly B resource."""
    s_max = int(log(R) / log(eta))
    B = (s_max + 1) * R
    seen = []
    for s in reversed(range(s_max + 1)):
        n = int(ceil((B / R) * eta ** s / (s + 1)))
        r = R * eta ** (-s)
        seen.extend(successive_halving(space, n, r, s, eta, rng,
                                       sample_configuration,
                                       run_then_return_val_loss, top_k))
    return min(seen, key=lambda cfg_loss_r: cfg_loss_r[1])[0]
```

# Random search

Random search chooses the trial set for a black-box hyper-parameter optimization by drawing
each configuration as an **independent uniform sample** from the configuration space — every
knob sampled independently from its own range (uniform, or uniform-in-log for knobs that span
orders of magnitude), over the same domain a grid would span. It keeps grid search's
conceptual simplicity, trivial parallelism, and reproducibility, but escapes grid's
exponential blowup, and is far more efficient in high-dimensional spaces whose response
function has **low effective dimension** — i.e. where only a few hyper-parameters actually
matter and which ones matter is unknown.

## Problem it solves

Minimize a response function `Psi(lambda) = mean_{x in X_valid} L(x; A_lambda(X_train))` over
a configuration space `Lambda`, where each evaluation of `Psi` means training and validating a
whole model (expensive), there are no gradients and no analytic form, the space mixes
continuous / integer / categorical knobs, and the budget is tens to a few hundred evaluations.
The deliverable is a strategy for choosing the trial points `{lambda^(1), ..., lambda^(S)}`.

## Key idea

Do not lay the trials on an aligned lattice; draw them **i.i.d. uniform**. The reasons:

- **Per-axis resolution (the projection argument).** A grid of `g` levels per dimension in `K`
  dimensions uses `N = g^K` trials but tests each single knob at only `g = N^{1/K}` distinct
  values, because all grid points share coordinates and collapse under projection. `N` i.i.d.
  points project (almost surely) to `N` distinct values on *every* axis. So on whichever
  low-dimensional subspace turns out to matter, i.i.d. sampling has the resolution of the full
  budget, while grid has only its `K`-th root — and i.i.d. gets this *without knowing* which
  subspace it is. Random search has the same efficiency on the relevant subspace as if it had
  been told the subspace and searched only it.

- **Dimension-free success probability.** Idealize the good region as a target of relative
  volume `v/V` placed in the unit hypercube. With `T` i.i.d. uniform trials, each independently
  misses with probability `(1 - v/V)`, so

  ```
  P(find target) = 1 - (1 - v/V)^T.
  ```

  For `v/V = 0.01`: `1 - 0.99^T`; `1 - 0.99^60 = 0.4528`, `T = 69` gives about
  50%, and the continuous 95% threshold is `ln(0.05)/ln(0.99) = 298.1`, so the
  first integer budget that reaches 95% is 299 trials. The ambient dimension `K`
  does not appear — only the target's relative volume. This is
  why random search "thrives on low effective dimensionality": a region that is wide in the
  irrelevant knobs and narrow in the few important ones still has findable relative volume in
  any number of dimensions.

- **Why grid is *especially* bad on low-effective-dimension targets.** Such targets are
  axis-aligned elongated rectangles (wide along irrelevant knobs, thin along important ones). A
  thin axis-aligned rectangle either threads through several collinear grid points at once
  (redundant, collapsing the effective sample size) or slips entirely between grid lines
  (catching none). I.i.d. points are never collinear, so each is an independent shot. (Grid
  only matches random when the target is *not* axis-aligned — which HPO targets essentially
  always are, since the important knobs are the coordinate axes.)

- **Log-scaling for spread-out knobs.** Knobs like learning rate, hidden-unit count, and
  regularization strength span orders of magnitude and the response is roughly flat per decade,
  so they are sampled uniformly in log space (exponentiate a uniform draw over
  `[log low, log high]`) to cover the *effect* evenly rather than the raw value.

## Why i.i.d. (not grid, not low-discrepancy)

The independence buys properties no aligned set has:

- **Anytime / extendable / fault-tolerant.** Stop whenever — what ran is a valid experiment;
  add trials later without recomputing; drop or restart a failed job freely. A grid breaks if a
  point is omitted; a Sobol / low-discrepancy set is built jointly, so adding or dropping a
  point disturbs it.
- **Analyzable.** `S` i.i.d. trials = `N` independent experiments of `s` trials (`sN <= S`),
  just by partitioning, so one run yields the whole distribution of best-of-`s` performance vs.
  experiment size (the random-experiment efficiency curve).
- **Low-discrepancy sequences (Sobol, Halton, Niederreiter, Latin hypercube)** can be a few
  percent better at small-to-mid budgets in low dimensions, but they are not i.i.d. (losing the
  above) and in the high-dimension-relative-to-budget regime of HPO they behave essentially
  like i.i.d. draws. Sobol is a refinement, not the headline.

## Honest reporting of the best model

Test error is not monotone in validation error and the validation set is finite, so reporting
the single best-validation trial's test score is optimistically biased when trials tie. This
does not change the proposal rule or the validation-based selection rule; it changes how the
selected model's test performance is reported. Treat the best model's test score `z` as a
Gaussian mixture over the `S` trials: component `s` has mean
`mu_s = Psi_test(lambda^(s))`, variance `sigma_s^2 = V_test(lambda^(s))`, and weight
`w_s = P(trial s wins)` where each `Z^(i) ~ N(Psi_valid(lambda^(i)), V_valid(lambda^(i)))`
(Bernoulli variance `V = Psi(1-Psi)/(|X|-1)` for 0/1 loss). Then

```
mu_z     = sum_s w_s * mu_s
sigma_z^2 = sum_s w_s * (mu_s^2 + sigma_s^2) - mu_z^2
```

(the mixture mean, and the law-of-total-variance for a mixture). The `w_s` are estimated by
simulation: draw hypothetical validation scores from those normals, count wins, normalize.

## Limitation (by design)

Random search is **non-adaptive**: it never consults the trials already run. Adaptive,
sequential, model-based / Bayesian methods that learn which dimensions matter as they go should
beat it — random search is the natural, reproducible, infrastructure-free baseline against
which to measure them.

## Final algorithm

```
for t = 1 .. T:
    lambda^(t) = ( independent draw of each knob from its range:
                   categorical -> uniform over choices
                   continuous  -> uniform[low, high]   (or exp(uniform[log low, log high]) if log-scaled)
                   integer     -> uniform integer        (or rounded log-uniform if log-scaled) )
    score^(t)  = Psi(lambda^(t))        # full-fidelity train + validate
return the lambda with the best validation objective

# Separately, report the chosen model's generalization with the Gaussian-mixture estimator above.
```

## Working code

Filling the `suggest` slot of the sequential search harness. Each call returns one independent
uniform draw at full fidelity; the history and remaining budget are intentionally unused,
because the method is non-adaptive and every trial is i.i.d.

```python
import numpy as np
from typing import Any, Dict, List, Tuple


class RandomSearch:
    """Random search: each trial is an independent uniform draw from the
    configuration space (uniform, or uniform-in-log for scale knobs). The
    history is never consulted -- i.i.d. draws probe every knob at the full
    budget's resolution regardless of which subspace matters."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)   # reproducible from the seed

    def suggest(
        self,
        space: "SearchSpace",
        history: List["Trial"],     # unused: non-adaptive
        budget_left: int,           # unused: every trial is i.i.d.
    ) -> Tuple[Dict[str, Any], float]:
        config = space.sample_uniform(self.rng)  # one independent per-knob draw
        return config, 1.0                       # full fidelity (single-fidelity baseline)
```

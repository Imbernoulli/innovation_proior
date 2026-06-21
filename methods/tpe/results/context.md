# Context: sequential black-box optimization of model hyper-parameters (circa 2011)

## Research question

Models that drive the state of the art — deep belief networks, stacked denoising
autoencoders, convolutional nets — carry anywhere from ten to fifty hyper-parameters:
learning rates, anneal schedules, layer sizes, pre-processing choices, regularization
penalties, contrastive-divergence settings, and so on. Their measured performance depends
sharply on getting these right, to the point where reported results are hard to reproduce and
the original investigations read "more of an art than a science." The job of setting them has
traditionally fallen to a human expert turning knobs by intuition.

The optimization this poses is unusually hostile. Each function evaluation means training a
full model and scoring it on a validation set, which can take hours on a GPU, so the entire
budget is tens to a few hundred trials — never thousands. The objective is a black box: we
get a scalar loss back from a configuration but no gradient and no structure we can see into.
The search space is *heterogeneous*: some variables are continuous (a learning rate), some
ordinal (the number of hidden units), some purely categorical (which pre-processing to use).
And, most distinctively, the space is *tree-structured*, i.e. conditional — a variable like
"number of units in the second layer" is only meaningful when a parent variable, "number of
layers," takes a value that makes a second layer exist. An optimizer here must not only pick
values for variables but simultaneously decide *which* variables are even in play for a given
configuration.

The question is how to build a sequential optimizer that uses the evidence from past trials
to concentrate future trials on promising regions, over a space that is mixed-type and
conditionally structured.

## Background

By 2011 the dominant way to set hyper-parameters is a mix of **manual search** and **grid
search**. Manual search is the expert turning knobs; it is effective in the hands of
specialists precisely because a person can be very efficient when only a few trials are
affordable, but it does not transfer, does not reproduce, and does not scale past a handful of
variables. Grid search picks a finite set of values for each of K variables and evaluates
every combination, so the number of trials is the product of the per-axis counts — it grows
exponentially in K and becomes infeasible past a few variables. Both are the state of practice
and the thing any new method is measured against.

An empirical observation reshapes how grid search should be judged. When the function from
hyper-parameters to validation performance is analyzed, for most data sets only a *few* of the
hyper-parameters actually matter — the response function has **low effective dimensionality**,
behaving roughly like a function of a small subset of its inputs — and *which* few matter
differs from data set to data set. Drawing a fixed number of trials at random instead of on a
grid gives every axis that many *distinct* values, so the axes that matter get covered far
better. This makes plain random sampling a surprisingly strong, and conceptually important,
reference point for this problem.

Several broader frames are on the table:

- **The configuration space as a generative process.** The cleanest way to describe a
  tree-structured, mixed-type space is a *generative procedure for drawing a valid sample*:
  first draw the number of layers from its prior, then, conditioned on that, draw each
  layer's parameters from their priors, and so on. Continuous variables come with priors over
  intervals (uniform, Gaussian, or log-uniform for quantities like learning rates that span
  decades); categorical variables with a prior probability vector. Sampling from this process
  is exactly what produces a valid configuration, conditional structure and all. Any history-
  blind baseline is "draw from this process and keep the best."

- **Sequential Model-Based Optimization (SMBO).** When evaluating the true objective `f` is
  expensive, the standard active-learning-style template (Hutter 2009; Hutter, Hoos &
  Leyton-Brown, LION-5 2011) is to keep a cheap *surrogate* model of `f` built from the
  observation history `H`, and on each round (i) maximize some acquisition criterion over the
  surrogate to choose the next point `x*` to try, (ii) pay to evaluate `f(x*)`, (iii) append
  `(x*, f(x*))` to `H`, and (iv) refit the surrogate. The expensive model fit happens once per
  round; everything else must be cheap by comparison. SMBO methods differ in two choices: what
  surrogate they build, and what acquisition criterion they maximize.

- **The acquisition criterion: Expected Improvement.** Given a probabilistic surrogate that,
  at a point `x`, gives a distribution over the unknown value `Y(x)`, and given the best value
  `y*` found so far, the Expected Improvement is the expected amount by which `x` would beat
  the incumbent, `EI(x) = E[max(y* - Y(x), 0)]` for a minimization problem. EI is the
  acquisition of choice (Jones, Schonlau & Welch 1998; Jones 2001) because it balances
  *exploitation* (regions whose predicted value is good) against *exploration* (regions where
  the surrogate is uncertain) automatically, with no hand-set target improvement, and it gives
  a clean stopping rule. Under a Gaussian surrogate with predictive mean `yhat(x)` and
  standard deviation `s(x)`, EI has a closed form `EI(x) = s(x)[u·Phi(u) + phi(u)]` with
  `u = (y* - yhat(x))/s(x)`, where `Phi` and `phi` are the standard normal CDF and PDF.

- **Probabilistic surrogates and their cost.** The classical SMBO surrogate is a **Gaussian
  process** (Mockus 1978; Rasmussen & Williams): a prior over functions, closed under
  conditioning, that returns an analytic posterior mean and variance and thus the predictive
  distribution EI needs.

- **Non-parametric density estimation.** Independently of optimization, there is a standard
  way to estimate a probability density from a finite set of samples without assuming a
  parametric family: the **Parzen-window / kernel density estimator** (Parzen 1962;
  Rosenblatt 1956), which places a kernel — typically a Gaussian of some bandwidth — at each
  observed sample and averages them. The bandwidth controls smoothness; too small overfits
  individual points, too large washes out structure. A density estimated this way can be built
  from local, coordinate-level notions of nearness rather than a single global Euclidean metric
  over the whole configuration vector.

## Baselines

These are the prior methods a new optimizer would be measured against and would react to.

**Manual search.** A human expert sets the hyper-parameters by intuition and iterative
refinement. *Core "algorithm":* none formal — domain knowledge plus a few exploratory fits.

**Grid search.** Choose a finite value set `L^(k)` for each variable `k` and evaluate every
combination; the trial count is `prod_k |L^(k)|`. *Core idea:* exhaustive coverage of a
discretized box.

**Random search** (Bergstra & Bengio 2011/2012). Draw configurations i.i.d. from the
generative prior over the space and keep the best one seen. *Core idea / algorithm:* repeat
`x ~ prior; y = f(x)` for the whole budget; return `argmin y`. On a response function with
low effective dimensionality it dominates grid search at equal budget, because each of the `S`
random trials gives `S` distinct values on every axis; it is trivially parallel and handles
mixed-type, conditional spaces because it simply samples the generative process.

**Gaussian-process SMBO with Expected Improvement** (Jones, Schonlau & Welch 1998; the GP
surrogate of Mockus 1978 / Rasmussen & Williams). Fit a GP to `H`, read off the predictive
mean `yhat(x)` and standard deviation `s(x)`, and choose the next point by maximizing the
closed-form EI `s(x)[u·Phi(u) + phi(u)]`, `u = (y* - yhat(x))/s(x)`, with `y*` the best
observed value. *Core idea:* a calibrated probabilistic surrogate of the objective whose
uncertainty drives exploration.

## Evaluation settings

The natural yardsticks that already exist for this problem:

- **Deep Belief Network hyper-parameter tuning** on the benchmark image data sets from
  Larochelle et al. (2007) — `mnist basic`, `mnist background images`, `mnist rotated
  background images` (MRBI), `convex`, `rectangles`, `rectangles images` — over a roughly
  32-variable configuration space (pre-processing, ZCA energy, classifier learning rate and
  anneal schedule, L2 penalty, number of layers from 1 to 3, batch size, and per-layer hidden
  units, weight-init scheme, contrastive-divergence epochs / learning rate / anneal schedule /
  sampling), with priors specified as uniform, Gaussian, log-uniform, quantized log-uniform,
  and categorical, including conditional per-layer groups. The metric is validation /
  test-set classification error; the budget is up to a couple hundred trials, each capped at
  about one hour of GPU time.
- **Multilayer perceptron tuning** on the Boston Housing regression task (506 points, 13
  scaled inputs, scalar target), over about ten hyper-parameters (learning rate, L1 and L2
  penalties, hidden-layer size, number of iterations, whether a PCA pre-processing is applied
  and its energy — the one conditional variable), scored by regression error.
- Protocol: every method is seeded with the same set of initial randomly sampled
  configurations, then run sequentially under a fixed trial budget; methods are compared by the
  quality of the best configuration found as a function of the number of trials. Multi-fidelity
  variants additionally vary how much of a full evaluation each trial pays for.

## Code framework

A new optimizer plugs into a generic sequential HPO harness that already exists. A
`SearchSpace` knows the per-variable priors (type, bounds, log-scale, categorical choices) and
can sample a valid configuration or clip one to bounds; a `Trial` records a configuration, the
validation score it earned, and the fidelity fraction it used; the driver loop repeatedly asks
the strategy for the next configuration, pays to evaluate it, and appends the result to the
history. Nothing about *how* the strategy turns a history into a proposal is settled — that
proposal rule is exactly what is to be designed — so the substrate is only the surrounding
machinery, with one empty slot.

```python
import numpy as np
from typing import Any, Dict, List, Tuple


class HParam:
    """One hyper-parameter's prior: name, type ('float'/'int'/'categorical'),
    low, high, log_scale, choices."""


class SearchSpace:
    """The generative prior over configurations (already exists)."""
    params: List[HParam]          # the per-variable priors
    dim: int

    def sample_uniform(self, rng) -> Dict[str, Any]:
        """Draw one valid configuration from the prior."""
        ...

    def clip(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Clip values back into their valid ranges."""
        ...


class Trial:
    config: Dict[str, Any]        # the configuration that was evaluated
    score: float                  # observed validation score (higher is better)
    budget: float                 # fidelity fraction used (1.0 = full evaluation)


class CustomHPOStrategy:
    """Proposes the next configuration to evaluate, given the history so far.
    Called once per round in a sequential loop."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # TODO: any per-strategy settings we decide to keep.

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        # TODO: the proposal rule we will design.
        #       Given the search space and the history of (config, score) trials,
        #       decide which configuration to evaluate next (and at what fidelity).
        #       Must stay cheap relative to one model fit, scale to dozens of
        #       mixed-type and conditional variables, and use the history.
        #       return config, fidelity
        pass


# existing sequential HPO driver the strategy plugs into
def run_search(objective, space, strategy, budget):
    history: List[Trial] = []
    for _ in range(budget):
        config, fidelity = strategy.suggest(space, history, budget - len(history))
        score = objective(config, fidelity)        # the expensive train+validate step
        history.append(Trial(config, score, fidelity))
    return max(history, key=lambda t: t.score).config
```

The driver supplies the history; `suggest()` is where the proposal rule will live.

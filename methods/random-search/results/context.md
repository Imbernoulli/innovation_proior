# Context: choosing trial sets for black-box hyper-parameter optimization

## Research question

Almost every learning algorithm `A` has knobs that are not fit by training — the
hyper-parameters `lambda` — and the algorithm only becomes a concrete predictor once they
are fixed: `f = A_lambda(X_train)`. What we actually want is the `lambda` that minimizes
generalization error,

```
lambda* = argmin_{lambda in Lambda}  E_{x ~ G_x} [ L(x; A_lambda(X_train)) ].
```

We cannot evaluate the expectation over the unknown data-generating distribution `G_x`, so in
practice it is replaced by a mean over a held-out validation set, giving a **response
function**

```
Psi(lambda) = mean_{x in X_valid} L(x; A_lambda(X_train)),     lambda* ~= argmin Psi(lambda).
```

This outer-loop optimization of `Psi` is the hyper-parameter optimization problem, and it is
brutal as an optimization target. `Psi` is a black box: a single evaluation means *training a
whole model and scoring it on validation*, which can take hours, so the budget is a few dozen
to a few hundred evaluations. There are no gradients of `Psi` with respect to `lambda`, no
analytic form, and essentially no prior knowledge of its shape — which dimensions of `lambda`
are sensitive, where the optimum sits, or how peaked it is. The configuration space `Lambda`
mixes continuous knobs that span orders of magnitude (learning rates, regularization
strengths), integer knobs (number of hidden units, number of layers), and categorical choices
(nonlinearity type, preprocessing). The precise goal is a strategy for choosing the set of
trial points `{lambda^(1), ..., lambda^(S)}` at which to evaluate `Psi`.

## Background

By this time, stochastic-gradient training of neural networks and deep belief networks is
driving rapid progress, and these models come with *many* hyper-parameters: learning rate and
its annealing schedule, weight-initialization scheme and scale, number of hidden units,
nonlinearity, regularization strength, preprocessing choice, minibatch size, and — for
multi-layer pretrained models — a separate block of pretraining knobs per layer, easily
reaching tens of hyper-parameters for a 3-layer model. The load-bearing facts about this
landscape:

- **The curse of dimensionality for exhaustive search (Bellman 1961).** If the space is
  indexed by `K` configuration variables and you pick a set of values `L^(k)` for each, the
  number of joint configurations is the product `prod_{k=1}^{K} |L^(k)|`, which grows
  exponentially in `K`. For a 32-dimensional configuration even two values per knob is
  `2^32 > 10^9` joint settings, each costing hours — far beyond any budget.

- **Low effective dimensionality of response functions.** A function can nominally depend on
  `K` variables yet be well approximated by a function of far fewer: if `f(x_1, x_2) ~=
  g(x_1)`, then `f` has effective dimension one even though it lives in two dimensions. The
  notion comes from the quasi-Monte-Carlo integration literature (Caflisch, Morokoff & Owen
  1997), where the efficiency of a point set is governed by how the integrand's variance is
  distributed across dimensions and low-order interactions rather than by the nominal
  dimension. Functions encountered in practice are typically far more sensitive to changes in
  some coordinates than others.

- **A diagnostic measurement of `Psi`'s sensitivity (Gaussian-process / ARD analysis).** One
  can fit a Gaussian process (Rasmussen & Williams 2006; Neal 1998) with a squared-exponential
  ("Gaussian") kernel that measures similarity between two values `a`, `b` of one
  hyper-parameter by `exp(-(a-b)^2 / l)`, with a separate positive length-scale `l` per
  hyper-parameter (a product/joint kernel across knobs), and choose each `l` by maximizing the
  marginal likelihood of observed `(lambda, Psi(lambda))` pairs. The reciprocal `1/l` is the
  **relevance** of that hyper-parameter — large relevance means `Psi` changes quickly along it.
  Run on neural-net response functions with seven hyper-parameters across a family of related
  classification data sets, this Automatic Relevance Determination shows two stubborn,
  reproducible properties of `Psi`:
  1. on any single data set, only a small fraction of the hyper-parameters carry most of the
     variation of `Psi` (effective dimension roughly 1 to 4 inside a 7-d problem), and
  2. *which* hyper-parameters are the important ones changes from data set to data set — the
     learning rate is consistently important, but whether the annealing rate, the
     regularization penalty, or the number of hidden units is the next most important depends
     on the data set.
  The measured relevance profile also makes clear that a surface with only one or two active
  coordinates is much easier to characterize than one whose variation is spread across several
  active coordinates.

These two properties together are the crux of the landscape: the target you are searching for
sits in a low-dimensional subspace of `Lambda`, but you do not get to know in advance which
subspace it is, and it moves when the data set changes.

## Baselines

The prior strategies a new trial-set strategy is measured against and reacts to.

**Grid search (e.g. LeCun et al. 1998b; Larochelle et al. 2007; and built into libsvm,
scikit-learn).** Choose a finite set of values `L^(k)` for each of the `K` knobs and evaluate
`Psi` at every element of the Cartesian product, `S = prod_k |L^(k)|` trials, returning the
best. Core appeal: conceptually trivial, easy to implement, trivially parallel (all trials
known up front), and reliable in 1-2 dimensions.

**Manual search / expert sequential tuning (as in Larochelle et al. 2007).** A human
alternates between fixing an architecture and hand-searching optimization knobs, coordinate-
descent style, using the validation feedback from earlier trials to choose later ones — often
combined with a coarse multi-resolution grid. Core appeal: it gives the researcher insight
into `Psi`, has no infrastructure overhead, and because it is adaptive (later trials benefit
from earlier ones) it can find good configurations in surprisingly few trials even in
high-dimensional spaces — the DBN tuning above averaged about 41 trials per data set across a
32-dimensional space.

**Low-discrepancy / quasi-random point sets (Sobol — Antonov & Saleev 1979; Halton 1960;
Niederreiter — Bratley et al. 1992; Latin hypercube — McKay et al. 1979).** Deterministic-ish
constructions that place points to match the uniform distribution as closely as possible
(minimize *discrepancy*: no clumps, no holes), often with the extra property that projections
onto subspaces stay low-discrepancy. In quasi-Monte-Carlo integration they reduce the variance
of finite-sample integral estimates faster than unstructured Monte Carlo.

**General global-optimization and early automated HPO methods (Nelder & Mead 1965 simplex;
Kirkpatrick et al. 1983 simulated annealing; Powell 1994 constrained optimization by linear
approximation; evolutionary strategies, Rechenberg 1973, Hansen et al. 2003; and sequential
model-based / Bayesian approaches, Hutter 2009; Hutter et al. 2011).** These are adaptive:
they use results already in hand to decide where to evaluate next, and the model-based ones in
particular can in principle weight dimensions by importance.

## Evaluation settings

The natural yardsticks already in use for judging a trial-set strategy:

- **Neural-net tuning on the Larochelle et al. (2007) benchmark suite** — eight image
  classification data sets built to contain many factors of variation: `mnist basic` and its
  variants (`mnist background images`, `mnist background random`, `mnist rotated`, `mnist
  rotated background images`), and the synthetic `rectangles`, `rectangles images`, and
  `convex` tasks. A single-hidden-layer network with seven hyper-parameters (number of hidden
  units, nonlinearity, weight-init scheme and scale, learning rate, learning-rate annealing,
  `L2` penalty), with the option of two preprocessing choices adding two more. The reference
  protocol is the grid search of Larochelle et al. (2007), which used on the order of 100
  trials per data set.
- **Deep-belief-network tuning on the same suite** — 1-, 2-, and 3-layer DBNs with 8 global
  hyper-parameters plus 8 per layer, up to 32 hyper-parameters for the 3-layer model, against
  the expert manual-plus-grid tuning of Larochelle et al. (2007), which averaged about 41
  trials per data set.
- **An artificial target-finding simulation** — locate a uniformly placed target occupying 1%
  of the volume of a unit hypercube, in 3-d and 5-d, with the target shaped either as a cube
  (full effective dimension) or as an axis-aligned elongated rectangle (low effective
  dimension), comparing how reliably standard trial-set designs locate it as a function of the
  number of trials.
- Metrics and protocol: validation-set loss / accuracy as the objective `Psi`; a
  generalization estimate for the *best* model that accounts for the uncertainty in which trial
  is truly best (treating the test score of the best model as a Gaussian-mixture random
  variable whose components are weighted by each trial's probability of being the winner, with
  Bernoulli variance `Psi(1-Psi)/(|X|-1)` for 0/1 loss); and an efficiency curve that partitions
  a completed batch into smaller batches to read off the distribution of best-of-`s` performance
  versus experiment size. Hyper-parameters that span orders of magnitude (learning rate, hidden
  units, penalty) are handled on a log scale.

## Code framework

A trial-set strategy plugs into a standard sequential hyper-parameter search harness: a search
space describing each knob, a loop that asks the strategy for the next configuration to
evaluate, trains and scores the model there, and records the trial. What the strategy *does* to
turn the space and the history into the next configuration is exactly the part to be designed,
so the substrate here is only the generic machinery that already exists: typed knob
descriptions, space-level utilities for constructing and validating configurations, a trial
record, and the outer loop. The single empty slot is the proposal rule.

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HParam:
    """One hyper-parameter: a continuous (optionally log-scaled), integer, or categorical knob."""
    name: str
    type: str                       # "float" | "int" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    choices: Optional[list] = None


@dataclass
class Trial:
    """Record of one evaluated configuration."""
    config: Dict[str, Any]
    score: float                    # validation score (higher is better)
    budget: float = 1.0             # fidelity fraction used (1.0 = full evaluation)


@dataclass
class SearchSpace:
    """The configuration space Lambda the search runs over."""
    params: List[HParam] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.params)

    def sample_uniform(self, rng: np.random.RandomState) -> Dict[str, Any]:
        """Construct one valid configuration, each knob filled in from its own
        declared range and type (a continuous knob over [low, high], possibly in
        log space; an integer knob; a categorical knob over its choices)."""
        ...  # config construction, already provided

    def clip(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Clip values back into the declared ranges."""
        ...  # bounds enforcement, already provided


class SearchStrategy:
    """A trial-set strategy. Called repeatedly; each call proposes the next
    configuration (and a fidelity in (0, 1]) to evaluate."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        # TODO: the proposal rule we will design.
        #       Given the search space (and, if we choose to use it, the
        #       history of past trials and the remaining budget), return the
        #       next configuration to evaluate and the fidelity to evaluate it at.
        pass


# existing sequential search loop the strategy plugs into
def search(space, evaluate, strategy, budget):
    history = []
    for _ in range(budget):
        config, fidelity = strategy.suggest(space, history, budget - len(history))
        score = evaluate(config, fidelity)         # train + validate at this configuration
        history.append(Trial(config, score, fidelity))
    return max(history, key=lambda t: t.score)
```

The outer loop hands the strategy the space, the trials so far, and the remaining budget;
`suggest` is where the proposal rule will live.

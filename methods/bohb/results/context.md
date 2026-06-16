# Context: hyperparameter optimization at scale (circa 2017-2018)

## Research question

The performance of a machine-learning model depends, often dramatically, on a vector of
hyperparameters `x` — learning-rate schedules, momentum, batch size, weight decay, dropout
rates, and increasingly architectural choices like the number and width of layers. Formally
we want `x* = argmin_{x in X} f(x)`, where `f(x)` is the validation loss of the model trained
with configuration `x`. The configuration space `X` mixes continuous, integer, and
categorical dimensions, and because training is stochastic (minibatch SGD, random
initialization, MCMC sampling, RL seeds), we never see `f(x)` directly but only a noisy
observation `y(x) = f(x) + eps`, `eps ~ N(0, sigma^2)`.

What makes this hard now, in a way it was not a few years ago, is cost. The best models take
days or weeks to train, so a single evaluation of `f` is enormously expensive. The total
budget a researcher can spend tuning is often not much larger than fully training a handful of
models. A practical optimizer therefore has to satisfy a demanding and partly conflicting set
of requirements at once:

1. **Strong anytime performance** — yield good configurations after a budget comparable to a
   few full trainings, not after hundreds.
2. **Strong final performance** — given a larger budget, return the best configuration in the
   space, which requires *guidance*, not luck.
3. **Effective use of parallel resources** — clusters and cloud are available; the method must
   exploit many workers near-linearly.
4. **Scalability** — handle anywhere from a few to many dozens of hyperparameters.
5. **Robustness and flexibility** — cope with very noisy objectives (deep RL), objectives
   sensitive to a few key knobs (probabilistic models), and all hyperparameter types (binary,
   categorical, integer, continuous).

Two further requirements bear specifically on whatever statistical model guides the search:
**simplicity** (few moving parts, easy to reimplement and verify) and **computational
efficiency** (the model-fitting and proposal cost must not dominate when evaluations are made
cheap). Each existing method below meets a subset of these; none meets all of them
simultaneously. Closing that gap — an optimizer that is fast early *and* converges to the best
configuration, while staying simple, scalable, parallel, and robust across hyperparameter
types — is the problem.

## Background

The field at this point splits into two largely separate lines of attack on `f`.

**Configuration selection — model-based / Bayesian optimization.** Treat `f` as a black box
and pick the next configuration intelligently. Bayesian optimization (BO) maintains a
probabilistic surrogate `p(f | D)` over the data `D = {(x_0, y_0), ..., (x_{i-1}, y_{i-1})}`
seen so far, and an *acquisition function* `a(x)` that trades exploration against
exploitation. A standard acquisition is the **expected improvement** (EI) over the incumbent
`alpha = min{y_0, ..., y_n}`,

```
a(x) = E[ max(0, alpha - f(x)) ] = integral max(0, alpha - f(x)) dp(f | D).
```

Each BO iteration (1) selects `x_new = argmax_x a(x)`, (2) evaluates `y_new = f(x_new) + eps`,
and (3) augments `D` and refits the surrogate. The dominant surrogate is the Gaussian process
(GP), prized for smooth, well-calibrated uncertainty. The crucial empirical fact about
black-box BO, observed repeatedly, is that **it behaves like random search early on**: with
few observations the surrogate is uninformative, so its first dozens of suggestions are
essentially random, and only later does the model start to pay off. Because every observation
here is a *full, expensive* training run, "later" can be very far away in wall-clock time.

**Configuration evaluation — multi-fidelity / bandit methods.** A different observation drives
the second line: for most ML objectives one can define *cheap approximations*
`tilde f(., b)` of `f` parameterized by a **budget** `b in [b_min, b_max]`, where
`tilde f(., b_max) = f` and smaller `b` gives a cheaper, noisier proxy. The budget can be
training epochs, the number of data points, the number of MCMC steps, or the number of RL
trials. The diagnostic phenomenon is that a configuration's quality is often *partially
revealed* at small `b`: poor configurations frequently look poor early, so spending full
budget on them is wasteful. This is what bandit-style methods exploit — pour resources into
configurations that look promising at low fidelity and cut the rest early. The catch, also
observed empirically, is that low-fidelity rankings are only *partially* faithful: a
configuration can look mediocre at small `b` and excellent at `b_max`, so any method that
commits hard to low-fidelity verdicts can discard the eventual winner.

These two lines barely talk to each other. Selection methods get guidance but treat `f` as an
all-or-nothing black box, paying full price per evaluation and starting blind. Evaluation
methods get the speed of cheap fidelities but, as detailed below, throw away the information
their many evaluations contain. The prevailing wisdom is that GP-based BO is the gold standard
for low-dimensional continuous problems, that GPs struggle on high-dimensional or mixed/
categorical spaces, and that simple resource-allocation heuristics are surprisingly strong for
small-to-medium budgets.

## Baselines

**Random search (Bergstra and Bengio, JMLR 2012).** Sample configurations uniformly from `X`.
Trivial, embarrassingly parallel, robust to the shape of `X`, and a strong baseline when only
a few hyperparameters matter. *Limitation:* it never uses what it has seen — every draw is
independent of all previous results — so to find a good configuration in a large space it must
get lucky, and its progress per evaluation does not improve over a run.

**Gaussian-process Bayesian optimization (Snoek et al., NIPS 2012; Hutter et al., LION 2011
for the random-forest variant SMAC).** GP-BO as above, with a GP surrogate. State of the art
on low-dimensional continuous problems. *Limitations:* fitting a GP is cubic in the number of
observations `O(|D|^3)` and scales poorly past a handful of dimensions; off-the-shelf kernels
do not apply to complex conditional or categorical spaces without bespoke engineering; and the
results depend sensitively on hyperpriors that themselves need careful, often
problem-dependent setting. Like all black-box BO it also starts cold, indistinguishable from
random search until the surrogate has enough data.

**Tree Parzen Estimator (TPE) (Bergstra, Bardenet, Bengio, and Kégl, NIPS 2011).** Rather than
modeling `p(y | x)`, model the *inputs* split by performance. Pick a quantile `gamma` of the
observed losses, set the threshold `alpha` so that `p(y < alpha) = gamma`, and fit two
densities over configuration space,

```
l(x) = p(x | y < alpha, D),     g(x) = p(x | y >= alpha, D),
```

`l` from the configurations that did well, `g` from the rest, each a Parzen/kernel density
estimator. TPE then proposes the configuration maximizing `l(x)/g(x)`. Bergstra et al. proved
this ratio is exactly the EI ordering: writing `p(x) = gamma·l(x) + (1-gamma)·g(x)`, the
expected improvement evaluates to

```
EI(x)  ∝  ( gamma + (g(x)/l(x))·(1 - gamma) )^{-1},
```

which is monotone decreasing in `g(x)/l(x)`, so maximizing EI is minimizing `g/l`, equivalently
maximizing `l/g`. Because it uses kernel density estimators, TPE handles mixed
continuous/discrete spaces naturally and its model-build cost is **linear** in `|D|`, not cubic —
directly addressing the GP scalability and
flexibility limitations. As originally formulated, TPE models the joint density as a *hierarchy
of one-dimensional* Parzen estimators (a product of 1-D pdfs), fitting each hyperparameter's
density independently. *Limitations:* it is still a black-box single-fidelity method — every
evaluation is a full training run, so it inherits the cold start and the per-evaluation
expense; and modeling each dimension on its own does not represent how hyperparameters
interact.

**SuccessiveHalving (SH) (Jamieson and Talwalkar, AISTATS 2016).** A bandit heuristic for
allocating a fixed budget across `n` candidate configurations. Evaluate all `n` on a small
budget, sort by performance, keep the best `1/eta`, multiply the budget by `eta`, and repeat
until the maximum budget is reached — so survivors get exponentially more resources. *The
"n versus B/n" limitation:* SH needs `n` as an input, and for a fixed total budget `B` the
budget per configuration is `~B/n`. There is no way to know in advance whether to prefer many
configurations each run briefly (large `n`, aggressive early stopping — right when quality
shows up early) or few configurations each run long (small `n`, conservative — right when
configurations only separate at large budgets). The correct choice depends on the unknown rate
at which configurations differentiate, and a wrong choice can stop the eventual winner
prematurely or waste the budget on too few candidates.

**Hyperband (HB) (Li, Jamieson, DeSalvo, Rostamizadeh, and Talwalkar, JMLR 2018;
arXiv:1603.06560).** Sidestep the `n`-versus-`B/n` choice by hedging over it: run SH for a
geometric ladder of `n` values. With `eta` and budgets in `[b_min, b_max]`, set
`s_max = floor(log_eta(b_max / b_min))`; then for each `s in {s_max, ..., 0}`, run an SH
"bracket" with ideal real-valued count
`n_ideal = ((s_max+1)/(s+1)) · eta^s`, rounded to an integer in an implementation, starting at
budget `r = b_max · eta^{-s}`. The bracket counts are chosen so every bracket consumes roughly
the same total resource `B = (s_max + 1)·b_max`: a bracket has about `s+1` rungs each costing
about `n_ideal·r`, and `(s+1)·n_ideal·r = (s_max+1)·b_max`. The most aggressive bracket (`s = s_max`,
largest `n`, smallest `r`) maximizes early stopping; the last bracket (`s = 0`) is plain
random search at full budget. HB inherits all of random search's robustness, scalability, and
flexibility, and adds strong anytime performance from cheap low-fidelity evaluations — it
typically beats random search and black-box BO at small-to-medium budgets. *Limitation:* it
draws the `n` configurations for every bracket **uniformly at random** and never conditions
those draws on the outcomes of past evaluations. So as the budget grows its advantage over
random search shrinks toward nothing, and its final-quality performance trails methods that
learn where good configurations lie.

The picture, then: the model-based methods (GP-BO, TPE) have guidance but no cheap fidelities
and a cold start; the multi-fidelity methods (SH, HB) have the cheap fidelities and the speed
but draw blind. Independent attempts to bring the two together existed concurrently — some
modeling the budget as just another GP input, some rebuilding a model from scratch inside each
SH run — but each inherited the GP limitations above or discarded information across
iterations.

## Evaluation settings

The yardsticks already in use at the time, on which any HPO method would naturally be measured
(the datasets, metrics, and protocol all pre-exist):

- **Synthetic mixed-space "counting ones"** — `f(x) = -( sum of N_cat binary x_i + sum over
  N_cont continuous x_j of E_{X ~ Bernoulli(x_j)}[X] )`, scalable to arbitrary dimension with
  both categorical and continuous variables; budget is the number of Bernoulli samples used to
  approximate the expectation. A controlled stress test for high-dimensional mixed spaces.
- **Support vector machine on MNIST** — tune the RBF-kernel `C` and `gamma`; budget is the
  fraction of training data, from `1/512` of the data up to the full set. A low-dimensional
  continuous problem (the regime where GP-BO is known to excel).
- **Feed-forward nets on OpenML datasets** (Adult, Higgs, Letter, MNIST, Optdigits, Poker) —
  six hyperparameters: initial learning rate, batch size, dropout, exponential learning-rate
  decay, number of layers, units per layer (ranges in Table; several log-scaled); budget is
  training time/epochs. Evaluated via random-forest surrogates fit to offline data so that
  hundreds of optimizer runs are affordable.
- **Bayesian neural networks** trained with SGHMC on UCI regression (Boston housing, protein
  structure) — tune step length, burn-in, units per layer, momentum decay; budget is the
  number of MCMC steps (`500` to `10000`); metric is validation negative log-likelihood.
- **Deep RL** — eight PPO hyperparameters on cartpole swing-up; budget is the number of trials
  (seeds); metric is episodes-to-convergence.
- **Convolutional net on CIFAR-10** — a residual network with Shake-Shake and Cutout; tune
  learning rate, momentum, weight decay, batch size; budget is training epochs.

The standard reporting protocol: identical configuration spaces across optimizers; the
*immediate regret* `|f(x_inc) - f(x*)|` of the incumbent `x_inc` as a function of wall-clock
time or budget consumed; mean and standard error over many independent runs; `eta = 3` for the
bandit methods as recommended for Hyperband.

## Code framework

The optimizer plugs into a standard ask/tell hyperparameter-search harness. The harness owns
the configuration space (which knows each hyperparameter's type, range, and whether it is
log-scaled, and can sample uniformly and encode a configuration into a numeric vector), and a
worker pool that, given a configuration and a budget/fidelity, trains the model and returns a
noisy validation loss. An outer driver repeatedly asks the strategy for the next configuration
(and the fidelity to evaluate it at), dispatches the evaluation, and tells the strategy the
result. What the strategy does internally — how it picks the configuration and the fidelity
from the accumulated history — is exactly what is to be designed; the substrate below is only
the generic machinery that already exists.

```python
import numpy as np


class ConfigSpace:
    """Pre-existing description of the search space X. Knows each hyperparameter's
    type ('float' / 'int' / 'categorical'), bounds, and log-scaling."""
    def sample_uniform(self, rng):
        """Draw one configuration uniformly at random from X."""
        ...
    def to_array(self, config):
        """Encode a configuration into a numeric vector in the unit hypercube
        (log-scaled dims mapped through log; categoricals to indices)."""
        ...
    def clip(self, config):
        """Clip values back into valid ranges."""
        ...


class Trial:
    """One completed evaluation."""
    config: dict      # the hyperparameter configuration
    loss: float       # observed validation loss; lower is better
    budget: float     # fidelity used (b_max == full evaluation)


class SearchStrategy:
    """Proposes the next (configuration, fidelity) to evaluate, given the history.
    The internal policy is what we will design."""

    def __init__(self, space, seed=42):
        self.space = space
        self.rng = np.random.RandomState(seed)
        # TODO: any per-strategy state we decide to keep across calls.

    def suggest(self, history, budget_left):
        """history: list[Trial] from past evaluations.
           budget_left: remaining budget in full-fidelity units.
           returns (config: dict, fidelity: float in (0, 1])."""
        # TODO: the search policy we will design — decide, from the accumulated
        #       (config, loss, budget) history, which configuration to try next
        #       and at what fidelity.
        raise NotImplementedError


# existing ask/tell driver the strategy plugs into
def optimize(strategy, evaluate, total_budget):
    history = []
    budget_left = total_budget
    while budget_left > 0:
        config, fidelity = strategy.suggest(history, budget_left)   # ask
        loss = evaluate(config, fidelity)                           # train + validate
        history.append(Trial(config, loss, fidelity))               # tell
        budget_left -= fidelity
    return min(history, key=lambda t: t.loss)
```

The driver supplies the history and consumes one `(config, fidelity)` per call; `suggest` is
where the search policy will live.

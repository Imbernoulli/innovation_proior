# Context: general-purpose hyperparameter optimization (circa 2018-2021)

## Research question

Modern machine-learning systems live or die by their hyperparameters: the architecture, the
optimizer settings, the regularization pipeline. Each candidate setting can only be scored by
*training a model and measuring validation performance*, so the objective `f(x)` over the
configuration space `X` is a black box, expensive, and noisy — we never see `f(x)` directly,
only `y(x) = f(x) + noise`. The goal is to find `x* in argmin_x f(x)` (a minimizer of
validation loss, equivalently a maximizer of validation score) under a tight evaluation
budget.

What makes this hard in practice is not just expense but *generality*. A method that is meant
to be a default — something a practitioner reaches for without thinking — has to satisfy
several demands at once: (1) **strong anytime performance**, so it returns a decent
configuration early and keeps improving; (2) **strong final performance** when given a large
budget; (3) **effective use of parallel resources**, since modern compute is distributed; (4)
**scalability with the dimensionality** of the configuration space; and (5) **robustness and
flexibility**, including search spaces with mixed continuous, integer, ordinal, and
categorical dimensions. The configuration spaces that matter most — joint architecture-plus-
training search, tabular neural-architecture-search problems posed as high-dimensional
discrete HPO — are exactly the ones where existing defaults are observed to falter. The
problem is to build a single method that hits all of these demands robustly, not a method that
wins on one axis and quietly fails on another.

## Background

The dominant framing treats HPO as **black-box optimization** and splits into two families:
*model-free* search (random search, evolutionary algorithms) and *model-based* Bayesian
optimization (BO). Two structural ideas underpin everything that follows.

**Multi-fidelity evaluation.** For many learning problems, a *cheap approximation* of the
expensive objective exists: train for fewer epochs, on a subset of the data, for fewer MCMC
steps, for fewer trials. These cheap evaluations are noisier but correlated with the full
evaluation, so they can rule out hopeless regions at a fraction of the cost. The notion of a
*budget* (or *fidelity*) `b` formalizes this — `b_min` is the cheapest approximation, `b_max`
the true objective. A central empirical fact, observed across HPO benchmarks, is that this
correlation is **strong for some problems and weak or absent for others**: on a 2-dimensional
SVM surrogate the ranking of configurations is nearly identical across budgets, while on
tabular NAS benchmarks the performance at few epochs and many epochs is only loosely related.
Any multi-fidelity method has to cope with both regimes.

**Population-based evolutionary search.** Evolutionary algorithms have been used for black-box
HPO since the 1980s (Grefenstette 1986) and for designing neural architectures (Angeline et
al. 1994; Real et al. 2017; recently Regularized Evolution, Real et al. 2019, reached
state-of-the-art on ImageNet). They are *model-free* and *gradient-free*: they keep a
population of candidate solutions, perturb and recombine them with operators inspired by
biology, and keep the fitter offspring. Because they only ever compare function values, they
are indifferent to the type or smoothness of the search space, which makes them natural
candidates for rugged, discrete, high-dimensional spaces.

The prevailing wisdom by 2018 is that **model-based BO gives the best final performance** but
carries real costs in this setting. Gaussian-process models give well-calibrated uncertainty
but scale poorly with dimensionality, do not natively handle complex or discrete spaces, and
have model-fitting cost that grows roughly cubically in the number of observations.
Tree-based and density-based surrogates relax some of this but inherit the high-dimensional
and discrete-space difficulties. A model also needs a minimum number of observations — on the
order of `d+1` in a `d`-dimensional space — before it can say anything useful, so in high
dimensions a model-based method spends a long opening phase behaving like random search.

## Baselines

These are the prior methods a new general HPO method would be measured against and would react
to.

**Random Search (RS) (Bergstra & Bengio 2012).** Sample configurations independently and
uniformly from the space, evaluate each at full budget, keep the best. Embarrassingly simple,
trivially parallel, and a famously strong baseline when only a few hyperparameters matter.
**Gap:** it never uses what it has already seen — every sample is independent of the history —
so on problems where good regions could be inferred from past evaluations it wastes the entire
budget exploring blindly, and it spends full budget on every config including obviously bad
ones.

**Differential Evolution (DE) (Storn & Price 1997).** A population-based evolutionary
algorithm that is unusually effective for its simplicity. Maintain `N` individuals, each a
`D`-dimensional vector. In each generation, for each target individual `x_{i,g}`:
- *mutation (rand/1):* pick three distinct individuals `x_{r1}, x_{r2}, x_{r3}` and form a
  mutant by adding a scaled difference vector,
  `v_{i,g} = x_{r1,g} + F * (x_{r2,g} - x_{r3,g})`, with scaling factor `F in (0,1]`;
- *crossover (binomial):* build a trial `u` by copying each coordinate from `v` with
  probability `p` and from `x` otherwise, forcing at least one coordinate from `v` via a random
  index `j_rand`;
- *selection:* keep the better of target and trial — for a minimizer, replace `x` by `u` iff
  `f(u) <= f(x)`, else retain `x`.
The difference vector is the elegant part: when the population is spread out the perturbations
are large (exploration), and as it converges they shrink (exploitation), so the step scale
*self-adapts* to the population's spread. rand/1 requires `N >= 3` distinct parents. To handle
mixed data types robustly, DE can keep the population entirely continuous in the unit hypercube
`[0,1]^D` and decode to the original (possibly discrete) space only at evaluation time, which
preserves diversity that a directly-discrete population would lose (Awad et al. 2020).
**Gap:** classical DE is **single-fidelity** — every individual is evaluated at full budget,
so there is no cheap early triage and no strong anytime behavior; and its classical update is
*deferred*, injecting a winning offspring into the population only after the whole generation
has been evolved, which delays the moment a good solution can influence the search.

**Successive Halving (SH) (Jamieson & Talwalkar 2016).** A multi-fidelity scheme: sample `N`
configurations, evaluate them all at the lowest budget, keep the top `1/eta`, multiply the
budget by `eta`, and repeat up to the highest budget. It allocates exponentially more resource
to the configurations that survive, so the expensive full evaluation is paid for only a
handful of candidates. **Gap:** it needs the number of configurations `N` as an input and
exposes the **`n`-versus-`B/n` dilemma** — for a fixed total budget `B`, should one try many
configurations cheaply (large `N`, little resource each) or few configurations expensively?
The right answer depends on the unknown budget-correlation structure. Worse, SH can discard at
a low budget a configuration that would have been best at the high budget, precisely when low-
and high-fidelity performance are weakly correlated.

**Hyperband (HB) (Li et al. 2017).** Hedge the `n`-versus-`B/n` dilemma by running SH at
several starting budgets. With maximum resource `R` and discard factor `eta` (default 3), set
`s_max = floor(log_eta R)` and total per-bracket budget `B = (s_max + 1) R`; then for
`s = s_max, ..., 0` run one SH *bracket* with
`n = ceil( (B/R) * eta^s / (s+1) )` configurations at starting resource `r = R * eta^{-s}`,
halving inward as `n_i = floor(n * eta^{-i})`, `r_i = r * eta^i`, keeping the top
`floor(n_i / eta)` each rung. The most aggressive bracket `s = s_max` throws many
configurations at a tiny budget; the least aggressive `s = 0` is plain random search at full
budget. This makes HB provably at most a constant factor (`~ s_max + 1`) slower than random
search, while exploiting cheap fidelities when they help — giving it strong anytime
performance. **Gap:** every configuration HB ever evaluates comes from **uniform random
sampling**; it never uses the outcomes of past evaluations to bias where it looks next. On long
runs it is therefore overtaken by methods that learn from history, and its quality is capped by
how good random sampling is in that space.

**BOHB (Falkner et al. 2018).** Keep Hyperband's bracket schedule, but replace its random
sampling with a Bayesian-optimization model — a Tree-Parzen-Estimator that, once it has enough
observations at a budget, fits kernel-density estimates `l(x)` of well-performing
configurations and `g(x)` of poorly-performing ones and samples to maximize the ratio
`l(x)/g(x)`. This grafts BO's strong final performance onto HB's anytime behavior and is the
strongest off-the-shelf multi-fidelity optimizer in wide use. **Gap:** the model component
inherits BO's weaknesses — it does not scale gracefully to high dimensions, struggles with
discrete dimensions, needs roughly `d+1` observations before its model is informative (so in
high dimensions it behaves like random search for a long opening), and its model-fitting
overhead grows with the number of observations rather than staying flat. On high-dimensional
discrete problems and tabular NAS its advantage over plain Hyperband shrinks or vanishes.

**CMA-ES (Hansen & Ostermeier 2001).** An evolution strategy that adapts a full covariance
matrix of a Gaussian sampling distribution; very strong on smooth continuous problems. **Gap:**
the covariance adaptation is built for continuous spaces and scales quadratically in the
dimension; it is single-fidelity and not designed for mixed discrete spaces.

## Evaluation settings

The natural yardsticks already in use for multi-fidelity HPO and NAS:

- **Stochastic Counting Ones** — a toy benchmark (introduced with BOHB) with `N_cat` binary
  and `N_cont` continuous dimensions; minimize the negative of the sum of the binaries plus the
  expected sum of Bernoulli means, with the budget controlling the number of samples used to
  estimate each mean (hence the noise). Run at `N_cat = N_cont in {4, 8, 16, 32}` to probe
  scaling and mixed binary/continuous handling.
- **OpenML feed-forward surrogates** — six architectural/training hyperparameters of a
  feed-forward net on six OpenML datasets (Adult, Higgs, Letter, MNIST, Optdigits, Poker), with
  the budget being the number of training epochs; a random-forest surrogate gives cheap lookups.
- **SVM-on-MNIST surrogate** — a 2-dimensional space (regularization `C`, kernel width
  `gamma`), budget = fraction of the training set; a benchmark with strong budget correlation.
- **Bayesian neural network** — a two-layer fully-connected BNN trained with SGHMC, 5
  hyperparameters, budget = number of MCMC steps (`500` to `10000`), on Boston Housing and
  Protein Structure (an extremely noisy setting).
- **Reinforcement learning** — a PPO agent on the OpenAI Gym cartpole task, 7 hyperparameters,
  budget = number of trials; episodes-to-convergence as the score.
- **Tabular NAS benchmarks** — 13 benchmarks from NAS-Bench-101, NAS-Bench-1shot1,
  NAS-Bench-201, and NAS-HPO-Bench, posed as high-dimensional discrete HPO problems, with
  training-epoch length as the fidelity.
- **Protocol** — report validation regret (distance from the best known score) against the
  cumulative function-evaluation cost (an anytime curve), averaged over many runs; rank methods
  per timepoint and average the ranks across benchmarks. Standard hyperparameter values for all
  methods are taken from prior work, and the optimizer's own overhead is excluded from the cost
  axis.

## Code framework

A new method plugs into the same multi-fidelity HPO harness the baselines already use. The
search space exposes the existing primitives: each hyperparameter has a type, bounds, and
choices; a configuration can be encoded to and decoded from the unit hypercube `[0,1]^D`
(integers rounded, categoricals binned), random configurations can be sampled, and an
out-of-range vector can be repaired. An evaluator runs a configuration at a requested fidelity
and returns a score; a history of past `(config, score, fidelity)` trials accumulates. The
bandit-style budget ladder (`b_min`, `b_max`, `eta`) and its Successive-Halving spacing already
exist — that arithmetic is fixed. What is *not* settled is the search policy: which
configuration to propose next, and at which fidelity. That is the single empty slot.

```python
import numpy as np


class SearchSpace:
    """Existing search-space abstraction over mixed hyperparameter types.
    Encodes/decodes between the original space and the unit hypercube [0,1]^D."""

    def __init__(self, params):
        self.params = params              # list of HParam(name, type, low, high, log_scale, choices)
        self.dim = len(params)

    def sample_uniform(self, rng):        # a random valid configuration (dict)
        ...

    def clip(self, config):               # repair an out-of-range configuration
        ...

    def encode(self, config):             # config dict  -> vector in [0,1]^D
        ...

    def decode(self, vec):                # vector in [0,1]^D -> config dict (round int, bin categorical)
        ...


def successive_halving_spacing(b_min, b_max, eta, iteration):
    """The fixed Successive-Halving spacing that already exists.
    Returns per-rung counts and actual fidelities."""
    s_max = int(np.floor(np.log(b_max / b_min) / np.log(eta)))
    s = s_max - (iteration % (s_max + 1))
    n0 = int(np.ceil(((s_max + 1) / (s + 1)) * eta ** s))
    n_configs = [max(int(n0 * eta ** (-i)), 1) for i in range(s + 1)]
    fidelities = [b_max * eta ** (-s + i) for i in range(s + 1)]
    return n_configs, fidelities


class SearchPolicy:
    """The search policy to design: propose the next configuration and the fidelity
    to evaluate it at, given the search space, the budget ladder, and the history."""

    def __init__(self, space, b_min, b_max, eta=3, seed=42):
        self.space = space
        self.b_min, self.b_max, self.eta = b_min, b_max, eta
        self.rng = np.random.RandomState(seed)
        # TODO: any state the policy we design will need

    def suggest(self, history, budget_left):
        # TODO: the search policy we will design --
        #       choose the next configuration and the fidelity to evaluate it at,
        #       given the space, the budget ladder, and the trials seen so far.
        config = None
        fidelity = None
        return config, fidelity


# existing multi-fidelity evaluation loop the policy plugs into
def run_hpo(policy, evaluate, budget_left):
    history = []
    while budget_left > 0:
        config, fidelity = policy.suggest(history, budget_left)   # the slot we design
        score = evaluate(config, fidelity)                        # train + validate at this fidelity
        history.append((config, score, fidelity))
        budget_left -= fidelity
    return history
```

The harness supplies the encoding, the budget ladder, and the evaluator; `suggest` — *what to
try next and how cheaply* — is where the method will live.

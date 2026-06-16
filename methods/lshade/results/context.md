## Research question

I have a continuous objective `f: R^D -> R` that I can only probe by evaluation — no gradient,
no structure, just "give me a point, I'll tell you its cost." The landscapes I care about span
the whole zoo: smooth unimodal bowls, ill-conditioned valleys (Rosenbrock-style), highly
multimodal egg-cartons (Rastrigin, Ackley), and composite functions stitched from several of
these. Dimensionality runs from a handful up to a hundred. The budget is a fixed number of
function evaluations, so every wasted evaluation is gone for good.

The engine of choice for this regime is Differential Evolution (DE): a population-based search
whose mutation reuses the population's own scatter as its step size. It is simple and often
strong. But its behavior is governed by three control parameters — population size `N`, scaling
factor `F`, crossover rate `CR` — and the single most-cited fact about DE is that the *optimal*
settings of these are problem-dependent. The right `F`/`CR` for a smooth unimodal function are
wrong for a rugged multimodal one; the right `N` for a 10-D problem is wrong for 100-D; and the
three interact. In practice one re-tunes by hand for every new problem, which is exactly the cost
a good method should remove.

So the goal is a DE that *tunes itself during the run*: it should (1) need no per-problem
hand-tuning of `F` and `CR`; (2) learn good parameter values online from what has actually been
working on *this* problem so far; (3) be robust — a single unlucky generation that rewards a bad
parameter value must not be able to derail the whole search; (4) manage exploration-versus-
exploitation over the budget, broad early and focused late; and (5) add as few new meta-knobs as
possible, since replacing one hard-to-set parameter with three more is no progress. Each of the
methods below achieves part of this. None achieves all of it at once. Closing that gap is the
problem.

## Background

**The DE substrate.** A DE population is a set of real vectors `x_i = (x_{i,1},...,x_{i,D})`,
`i = 1..N`. A standard generation does three things per individual. *Mutation* builds a donor
vector by adding scaled population differences to a base vector; the canonical "DE/rand/1" form
is

```
v_i = x_{r1} + F * (x_{r2} - x_{r3}),   r1, r2, r3 distinct and != i,
```

with `F > 0` the scaling factor. *Binomial crossover* mixes the donor with the parent
coordinate-by-coordinate,

```
u_{i,j} = v_{i,j}  if rand[0,1) <= CR  or  j == j_rand,
          x_{i,j}  otherwise,
```

where `j_rand` is a uniformly chosen index in `[1,D]` that guarantees at least one coordinate
comes from the donor (so the trial never just copies the parent). *Selection* is a greedy
one-to-one replacement that keeps the better of parent and trial:

```
x_{i,G+1} = u_{i,G}  if f(u_{i,G}) <= f(x_{i,G}),  else  x_{i,G}.
```

The deep reason DE works is that the difference vector `x_{r2} - x_{r3}` is *self-scaling*: when
the population is spread out (early), the differences are large and the search explores; as the
population contracts toward an optimum, the differences shrink and the search refines — an
automatic exploration-to-exploitation transition with no explicit step-size schedule. The naming
convention `DE/base/k/cx` records the base vector, the number of difference vectors, and the
crossover type (e.g. `DE/rand/1/bin`).

**The known pain points of fixed parameters.** Empirically, `F`, `CR`, `N` are problem-dependent
in a way that is well documented across the DE literature. A higher `F` and larger `N` favor
exploration and resist premature convergence on multimodal landscapes but converge slowly; a
lower `F`, smaller `N`, and the right `CR` converge fast on unimodal or near-separable problems
but risk collapsing into a local optimum on rugged ones. Crucially `N` is not a free lunch in
either direction: small populations converge quickly but get trapped; large populations explore
widely but burn the evaluation budget making slow per-generation progress. The optimal `N`
depends on the problem, the dimensionality, and on `F`/`CR` themselves. This three-way coupling,
all problem-dependent, is the diagnostic finding that motivated more than a decade of work on
adaptive DE.

**Distributions for sampling parameters.** When parameters are sampled rather than fixed, two
distributions recur and matter. A Normal `randn(mu, sigma)` concentrates samples tightly around a
center — appropriate when you want a stable estimate that does not stray far. A Cauchy
`randc(mu, sigma)` has the same center but heavy tails, so it routinely produces samples far from
`mu`; this is the textbook tool for diversification — keeping a parameter from prematurely
collapsing onto one value, because the tail keeps proposing alternatives. The arithmetic mean of
a set is its plain average; for positive values, the **Lehmer mean**
`mean_L(S) = (sum_k S_k^2) / (sum_k S_k)` is a different summary that always sits at or above the
arithmetic mean and is pulled *upward* by the larger elements of `S` (it equals the arithmetic
mean only when all elements are equal). Both are pre-existing tools.

**Population resizing as a deterministic schedule.** Because adaptive population *resizing*
schemes (e.g. GAVaPS) were found to trade the single parameter `N` for several meta-level
parameters that are themselves hard to set, the field moved toward *deterministic*, monotone
population-size schedules that change `N` on a predetermined rule rather than reacting to the
search state. Representative examples increase the population (IPOP-CMA-ES) or decrease it
(Dynamic Population Size Reduction, which halves `N` at fixed intervals; Simple Variable
Population Sizing (SVPS, Laredo et al.), a general framework whose reduction-curve shape is set by
parameters `tau`, `rho` together with the initial and final sizes). These deterministic rules
were found to be highly effective and far simpler to use than reactive resizing.

## Baselines

These are the prior methods a new self-adaptive DE would be measured against and would react to.

**Classical DE (Storn & Price, 1997).** The substrate above with *fixed* `N`, `F`, `CR`
(common defaults `F = 0.5`, `CR = 0.9`, `N` a few times `D`). Strong and simple. **Gap:** the
three parameters are static and problem-dependent; performance swings with the landscape and
there is no mechanism to discover good settings during a run — the user must tune by hand for
each new problem.

**jDE (Brest et al., 2006).** Encode the parameters *on the individuals*: each `x_i` carries its
own `F_i, CR_i` (initialized `F_i = 0.5, CR_i = 0.9`), inherited by its offspring, and each is
randomly re-rolled within a fixed range with a small probability; a re-rolled value is retained
only if the resulting trial wins selection. So good parameter values propagate by survival.
A variant (dynNP-jDE) additionally halves the population periodically. **Gap:** the "what worked"
signal is diffuse — spread across the individuals and entangled with selection — with no explicit
record of *which* parameter values caused success, so adaptation is indirect and slow to react.

**SaDE (Qin et al., 2009).** Keep a small memory and a pool of mutation strategies; each
generation pick a strategy per individual with probability biased by how often each strategy
recently succeeded. For crossover, sample `CR_i ~ randn(CRm_k, 0.1)` where `CRm_k` is the
*median* of recently successful `CR` values for strategy `k`; `F` is drawn from `randn(0.5, 0.3)`
and not adapted. **Gap:** adaptation tracks a single central tendency (a median) per strategy and
re-samples around it, so it represents the "good region" of parameter space by one point — it
cannot hold several distinct good values at once, and it adapts only `CR`, not `F`.

**EPSDE (Mallipeddi et al., 2011) and CoDE (Wang et al., 2011).** Pool-based ensembles. EPSDE
keeps three pools (mutation strategy; `F` values `0.4..0.9` step `0.1`; `CR` values `0.1..0.9`
step `0.1`), assigns a combination to each individual, lets successful combinations be inherited
and reinitializes failed ones. CoDE does not adapt at all: it draws random combinations of three
hand-chosen strategies with three hand-chosen `[F,CR]` pairs (`[1.0,0.1]`, `[1.0,0.9]`,
`[0.8,0.2]`), generating three trials per individual. **Gap:** both restrict parameters to a
coarse, hand-built discrete menu rather than learning continuous values from the run's own
success history; the granularity and the menu contents are themselves design choices that may not
fit a given problem.

**JADE (Zhang & Sanderson, 2009).** The most directly relevant ancestor; it contributes three
pieces that self-adaptive DE variants can keep.
(1) **DE/current-to-pbest/1 mutation**, a less-greedy generalization of current-to-best/1:

```
v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_{r1} - x_{r2}),
```

where `x_pbest` is drawn uniformly from the top `N*p` individuals (`p in (0,1]`) rather than from
the single best, so a small `p` is greedy and a larger `p` is exploratory. Pulling toward a *set*
of good solutions instead of the single best was introduced because current-to-best/1 directs
everyone at the one incumbent best and converges prematurely on multimodal problems.
(2) **An optional external archive** `A`: parents that *lose* selection are not discarded but
stored, and when the archive is in use, `x_{r2}` in the mutation is drawn from `P ∪ A` (population
union archive). The archived losers carry information about recently abandoned directions, adding
diversity to the difference vector. The archive is capped at `|A| = |P|`; when full, random
elements are deleted.
(3) **Adaptive `(mu_CR, mu_F)`.** Each individual samples `CR_i = randn(mu_CR, 0.1)` (clamped to
`[0,1]`) and `F_i = randc(mu_F, 0.1)` (truncated to `1` if above, *re-sampled* if `<= 0`). At
each generation the successful values `S_CR, S_F` update the centers with learning rate `c = 0.1`:

```
mu_CR <- (1 - c) * mu_CR + c * mean_A(S_CR)        (arithmetic mean)
mu_F  <- (1 - c) * mu_F  + c * mean_L(S_F)         (Lehmer mean)
```

The Lehmer mean for `F` deliberately upweights larger successful `F` values to keep the mutation
magnitude from decaying; the Cauchy distribution for `F` keeps `F` diverse. **Gap:** all sampling
is guided by a *single* pair `(mu_CR, mu_F)`. Because selection is probabilistic, a given
generation's `S_CR, S_F` can contain poor values; when they do, the single centers drift toward
those poor values and the *entire* population's sampling degrades the next generation. The
adaptation carries only one piece of state, so a single misleading generation moves the only
thing the whole population samples from.

Across all of these adaptive variants, one control parameter remains untouched: the population
size `N` is held fixed for the entire run. The exploration-versus-refinement compromise that `N`
forces (broad early, fine-grained late) is therefore the same as in classical DE, regardless of
how cleverly `F` and `CR` are adapted.

## Evaluation settings

The standard yardsticks already in use for real-parameter single-objective optimization:

- **The CEC benchmark suites** (CEC2005, CEC2013, CEC2014). The CEC2014 suite is 30 functions on
  the search box `[-100, 100]^D`: `F1..F3` unimodal, `F4..F16` simple multimodal, `F17..F22`
  hybrid functions (variables partitioned into 3-5 groups each evaluated by a different
  sub-function — a form of partial separability meant to mimic real-world structure), and
  `F23..F30` composite functions stitching several test problems into one landscape.
- **Classic textbook functions** also used in the DE literature: Rastrigin (multimodal,
  `[-5.12, 5.12]^D`), Rosenbrock (ill-conditioned valley), Ackley (multimodal,
  `[-32.768, 32.768]^D`), Sphere, etc., all with known global minima.
- **Protocol.** Dimensionalities `D in {10, 30, 50, 100}`; a fixed evaluation budget of
  `D * 10,000` function evaluations per run; results aggregated over many independent runs (e.g.
  51); the reported quantity is the error between the best fitness found and the known optimum,
  with errors below `1e-8` treated as zero. Metrics are the final best fitness (lower is better)
  and how quickly the run reaches near-final fitness. Automatic configuration tools (e.g.
  ParamILS) are available for tuning an algorithm's own meta-parameters on a *training* subset of
  functions disjoint from the test set.

## Code framework

A new self-adaptive DE plugs into the same generic population-based black-box optimization harness
the baselines use. What is *not* yet settled — and is exactly what is to be designed — is how each
individual's reproduction is parameterized and how the population is managed across the run. So the
substrate is only the generic machinery that already exists: a population of real vectors with
fitnesses, an evaluation oracle, an outer generation loop, a generic reproduce-then-select step,
and bookkeeping for the best-so-far and a per-generation fitness history. The single empty slot is
the reproduction-and-population-management policy itself.

```python
import numpy as np


def initialize_population(n, dim, lo, hi):
    """Uniform-random real-vector population in the box [lo, hi]^dim."""
    return [lo + np.random.rand(dim) * (hi - lo) for _ in range(n)]


def binomial_crossover(parent, donor, cr, dim):
    """Standard DE binomial crossover: each coord from donor w.p. cr, with one
    guaranteed donor coord (j_rand). cr is supplied by the policy under design."""
    j_rand = np.random.randint(dim)
    trial = parent.copy()
    for j in range(dim):
        if np.random.rand() < cr or j == j_rand:
            trial[j] = donor[j]
    return trial


def repair_to_bounds(parent, donor, lo, hi):
    """Existing DE bound handling: move an infeasible donor coordinate halfway
    back toward the parent's coordinate and the violated bound."""
    donor = np.where(donor < lo, (lo + parent) / 2.0, donor)
    donor = np.where(donor > hi, (hi + parent) / 2.0, donor)
    return donor


def run_evolution(evaluate, dim, lo, hi, pop_size, max_nfes, seed):
    """Generic population-based black-box optimizer over a fixed evaluation budget.

    Owns the population, the evaluation budget, and the generation loop.
    The reproduction parameterization (how each individual's donor and crossover
    are formed) and how the population evolves in size/composition over the run
    are NOT decided here -- that is the policy to be designed.
    """
    np.random.seed(seed)
    pop = initialize_population(pop_size, dim, lo, hi)
    fitness = [evaluate(ind) for ind in pop]
    nfes = pop_size
    fitness_history = []

    while nfes < max_nfes:
        for i in range(len(pop)):
            # TODO: the reproduction-and-population-management policy we will design.
            #       Given the population, the fitnesses, and the run's progress,
            #       form a trial vector for individual i and decide what survives.
            #       donor  = <policy>(pop, fitness, i)
            #       donor  = repair_to_bounds(pop[i], donor, lo, hi)
            #       trial  = binomial_crossover(pop[i], donor, <cr>, dim)
            #       keep the better of pop[i] and trial; nfes += 1
            pass
        fitness_history.append(min(fitness))

    best = pop[int(np.argmin(fitness))]
    return best, fitness_history
```

The harness supplies the population, the oracle, and the loop; the empty slot is where the
self-adapting reproduction rule and the population schedule will live.

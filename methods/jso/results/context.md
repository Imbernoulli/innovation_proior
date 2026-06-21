## Research question

I have a continuous objective `f: R^D -> R` that I can only probe by evaluation — no gradient, no
structure, just "give me a point, I'll tell you its cost." The landscapes span the zoo: smooth
unimodal bowls, ill-conditioned valleys, highly multimodal egg-cartons, hybrid and composite
functions. Dimensionality runs from 10 to 100, and the budget is a fixed number of function
evaluations, so every wasted evaluation is gone for good.

The strongest off-the-shelf engine for this regime is the success-history adaptive DE family —
SHADE and its linear-population-reduction extension L-SHADE, the CEC 2014 winner. L-SHADE already
self-tunes `F` and `CR` from a memory of what has been working, mutates with current-to-pbest/1 plus
an external archive, and shrinks the population linearly over the budget. The question is how to
build on the L-SHADE family to compete on the newer, harder benchmark suites.

## Background

**The L-SHADE substrate.** A population of `N` real vectors. Each generation, individual `i` samples
`F_i` from a Cauchy centered on a memory slot and `CR_i` from a Normal centered on a (possibly
different) slot; forms the donor by current-to-pbest/1,
`v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)` with `x_pbest` from the top `N*p`, `x_r1` from
`P`, `x_r2` from `P ∪ A` (the external archive of losing parents); binomial-crosses to a trial;
greedily keeps the better of trial and parent. Winners' `(F_i, CR_i)`, weighted by their fitness
improvement, are summarized into one memory slot per generation (round-robin): the **weighted Lehmer
mean** `mean_WL(S) = (Σ w_k S_k^2)/(Σ w_k S_k)` for both `F` and `CR` (with a terminal-zero rule for
`CR`); the Lehmer mean sits at or above the weighted arithmetic mean and is pulled up by larger
values. The population shrinks linearly, `N -> N_min` over the budget, deleting the worst.

**iL-SHADE refinements (Brest et al., 2016).** Several changes to L-SHADE, all aimed at the early
run. (1) Memory initialization is biased high: all `M_CR` slots start at `0.8` (not `0.5`) and the
`M_F` slots start at `0.5`, propagating higher `CR` early, when a higher `CR` (more donor
coordinates) and the implied higher `F` help. (2) One memory slot (the last) is *fixed* at
`M_F = M_CR = 0.9` and never updated, a permanent reservoir of an aggressive setting that some
fraction of individuals always sample. (3) Very high `F` and very low `CR` are forbidden early:
`CR` is floored (`CR >= 0.5` for the first quarter of the budget, `>= 0.25` for the first half) and
`F` is capped (`F <= 0.7` for the first quarter, `<= 0.8` for the first half, `<= 0.9` for the first
three quarters). (4) The memory update averages the new
weighted-Lehmer summary with the *old* slot value, `M[k] <- (mean_WL(S) + M[k]_old)/2`, so a single
generation cannot fully overwrite a slot. The `p` for current-to-pbest is also made to *decrease*
linearly over the run, from a larger early value to a smaller late value.

**Distributions and means.** Cauchy `randc(mu, 0.1)` (heavy-tailed, keeps `F` diverse, resample if
`<= 0`, truncate to `1`); Normal `randn(mu, 0.1)` (tight, keeps `CR` near its center, clamp to
`[0,1]`); weighted Lehmer mean as above. The `p` fraction sets current-to-pbest greediness; the
archive caps at `|A| = N`, random deletion on overflow; `N_min = 4` because current-to-pbest/1 needs
four distinct individuals.

## Baselines

- **L-SHADE (Tanabe & Fukunaga, CEC 2014).** Success-history adaptive DE + linear population
  reduction; CEC 2014 winner. Memory centers start at `0.5`; a single `F` scales both the
  pbest-attraction term and the random-difference term.
- **iL-SHADE (Brest et al., 2016).** L-SHADE plus the four early-run refinements above; mutation is
  current-to-pbest/1 with one `F` for both difference terms.
- **JADE (Zhang & Sanderson, 2009).** The ancestor: current-to-pbest/1, the archive, single-center
  `(mu_F, mu_CR)` adaptation; no population reduction.

## Evaluation settings

The CEC 2017 suite (30 functions, `[-100,100]^D`, `D in {10,30,50,100}`), classic textbook functions
(Rastrigin, Rosenbrock, Ackley), a fixed budget of `D*10,000` evaluations, errors below `1e-8`
treated as zero, results aggregated over many runs; metrics are final best error (lower better) and
convergence speed.

## Code framework

The method plugs into the same generic population-based black-box optimization harness the baselines
use; the empty slot is the reproduction-and-population-management policy. The harness owns the
population, the evaluation oracle, the generation loop, and the best-so-far / fitness-history
bookkeeping.

```python
import numpy as np


def initialize_population(n, dim, lo, hi):
    """Uniform-random real-vector population in the box [lo, hi]^dim."""
    return [lo + np.random.rand(dim) * (hi - lo) for _ in range(n)]


def binomial_crossover(parent, donor, cr, dim):
    """Standard DE binomial crossover: each coord from donor w.p. cr, one
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

    Owns the population, the budget, and the loop. The reproduction parameterization
    and the population schedule are NOT decided here -- that is the policy to design.
    """
    np.random.seed(seed)
    pop = initialize_population(pop_size, dim, lo, hi)
    fitness = [evaluate(ind) for ind in pop]
    nfes = pop_size
    fitness_history = []

    while nfes < max_nfes:
        for i in range(len(pop)):
            # TODO: the reproduction-and-population-management policy.
            pass
        fitness_history.append(min(fitness))

    best = pop[int(np.argmin(fitness))]
    return best, fitness_history
```

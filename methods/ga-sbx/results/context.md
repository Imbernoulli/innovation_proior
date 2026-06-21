# Context: real-coded genetic search for continuous optimization (early-to-mid 1990s)

## Research question

Genetic algorithms have a strong track record on problems with a *discrete* search
space, and the standard story for *why* they work is the building-block hypothesis:
a single-point crossover on binary strings recombines short, high-fitness substrings
("schemata") from two parents into a child, and over generations the good substrings
accumulate. The crossover operator is the main search engine, and its power is tied to
how single-point crossover propagates these building blocks.

The setting here is one in which the variables are *continuous reals* over a box
`[x_lo, x_hi]^n` and the objective `f: R^n -> R` is a black box we only query. The
established move is to binary-encode each real variable in a fixed number of bits and
run an ordinary binary GA. That encoding has four well-known characteristics: (1) fixed,
finite precision — the achievable resolution is set by the bit length; (2) a fixed
mapping from strings to reals; (3) the **Hamming-cliff** structure — adjacent real values
can be far apart in Hamming distance (e.g. `0111` and `1000` are neighbours in value but
differ in every bit), so a small move in real space corresponds to flipping many bits at
once; and (4) building-block / schema processing defined over discrete alleles.

The question is how to run a crossover-driven GA **directly on the real-valued
variables**, composing a real-coded recombination operator with real-valued mutation and
a selection rule so the whole evolutionary loop stays in real space end to end.

## Background

**The genetic loop.** A generational GA holds a population of candidate solutions and
repeats: *selection* (pick parents biased toward higher fitness), *crossover*
(recombine pairs of parents into offspring), *mutation* (perturb offspring), then
replace the population with the offspring and iterate. For real-coded GAs the individual
is just a vector of floats and each operator must act on floats.

**Single-point binary crossover on the decoded reals.** Two parent bitstrings are cut at
a random site and the tails swapped. Decode a string as `x = B·2^k + A` where `A` is the
right `k` bits and `B` the left bits; the two children keep each parent's `B` and swap the
`A` parts. A short computation on the decoded values shows the arithmetic mean of the two
decoded children equals the mean of the two parents: `(y1 + y2)/2 = (x1 + x2)/2`. Under a
linear bit-to-real mapping this carries over to the real values, so the children sit
symmetrically about the parents' midpoint.

**Offspring distribution.** A crossover can be characterized by the distribution of
children it produces from two fixed parents. Single-point crossover has an analyzable
distribution on decoded child locations: it tends to keep offspring near the parental
values while still sometimes moving outside the parental interval, a qualitatively
different shape from a flat draw over a widened interval.

**Selection pressure without fitness scaling.** Tournament selection (Goldberg & Deb,
*Foundations of Genetic Algorithms*, 1991) repeatedly samples a small group of `t`
individuals uniformly at random and keeps the best; doing this `k` times yields `k`
parents. Larger `t` raises selection pressure (greedier), and the scheme needs no
fitness normalization or sorting — it works off pairwise fitness comparisons alone. In a
toolbox with weighted fitness values, the same comparison mechanism handles whether the
task is minimization or maximization.

**Real-coded mutation as a local perturbation.** For real variables a mutation operator
perturbs a coordinate to a nearby value, with small perturbations more likely than large
ones, while respecting the box bounds. The mutation probability per variable was studied
in the binary setting: a per-bit rate of `1/L` (with `L` the total string length) was
found to perform best (Mühlenbein, 1992), giving roughly one mutation per individual on
average.

## Baselines

**Single-point binary crossover (Holland; Goldberg, 1989).** Cut two bitstrings at a
random site, swap tails. It provides provable building-block propagation (Holland's schema
theorem) and the nonuniform decoded-child profile above. Applied to continuous problems,
it acts on a binary *encoding* of the reals, carrying the four encoding characteristics
above: fixed precision, fixed mapping, Hamming cliffs, and schema processing over discrete
alleles.

**Linear / heuristic crossover (Wright, 1991).** Operates on reals directly: from two
parents `p1, p2` it forms the three points `(p1+p2)/2`, `1.5·p1 − 0.5·p2`, and
`−0.5·p1 + 1.5·p2`, and keeps the best two. The operator is deterministic given the parent
pair: three candidate children.

**Blend crossover BLX-α (Eshelman & Schaffer, 1993).** Also real-coded. For parents
`p1 < p2`, sample the child uniformly from the widened interval
`[p1 − α(p2 − p1), p2 + α(p2 − p1)]`. With `α = 0` this draws uniformly *between* the
parents; `α = 0.5` (BLX-0.5) was reported best across a range of problems. BLX-0.0
preserves the "interval schemata" (contiguous regions common to both parents), and its
interval width scales with the parent separation. The offspring density is flat over the
blend interval — every point in the sampled box is equally likely — and the `α` knob
sets the box width.

**Evolution-strategy recombination (Schwefel; Rechenberg).** Real-valued from the
outset, with Gaussian mutation as the primary search operator; recombination is usually
discrete (pick one parent's value) or intermediate (average the two parents). Discrete
and intermediate recombination are deterministic per coordinate, producing one point;
ES relies on mutation to explore.

## Evaluation settings

The natural yardsticks already in use for continuous black-box GA studies, all
minimization with a known global optimum, run with a fixed population over a fixed number
of generations and compared at identical initialization:

- Standard real-parameter test functions over a box `[lo, hi]^n` per dimension:
  the **sphere / ellipsoidal** family (unimodal, separable); **Rosenbrock**
  (ill-conditioned curved valley, non-separable); **Rastrigin** (highly multimodal,
  a quadratic bowl plus a cosine ripple with many local minima); **Ackley** (multimodal
  with a single deep global basin and a near-flat outer region); **Schwefel**-type sums.
- Dimensionality is swept (e.g. tens of variables up to ~100) to probe scaling.
- Metrics are the best objective value reached and how many generations it takes to get
  near the final value (a convergence-speed measure). Robustness is read off repeated
  runs from different random initial populations with a common seed protocol.
- GA hyperparameters that have to be set: population size, number of generations,
  crossover probability, and per-variable mutation probability.

## Code framework

The harness is the standard real-coded generational GA, and the population, evaluation,
and the outer loop already exist. What is *not* settled is the three genetic operators
themselves — how to recombine two real vectors, how to perturb one real vector, and how
to pick parents — so those are left as empty slots. The substrate uses the existing
toolbox primitives (an individual is a list of floats carrying a `.fitness.values`
tuple; the population, evaluation map, and clone/replace machinery are provided).

```python
import random
import numpy as np
from typing import Callable, Tuple

# existing toolbox primitives: individual = list[float] with .fitness.values,
# population construction, evaluation, cloning, and bound clipping all already exist.


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Pick k parents from the population, biased toward better fitness."""
    # TODO: the selection operator we will choose.
    pass


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Recombine two real-valued parents into two offspring, in place."""
    # TODO: the real-coded recombination operator we will design.
    pass


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Perturb a real-valued individual within [lo, hi], in place."""
    # TODO: the real-coded mutation operator we will design.
    pass


def run_evolution(
    evaluate_func: Callable,
    dim: int,
    lo: float,
    hi: float,
    pop_size: int,
    n_generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
) -> Tuple[list, list]:
    """Generational GA loop: select -> crossover -> mutate -> evaluate -> replace."""
    random.seed(seed)
    np.random.seed(seed)

    pop = init_population(pop_size, dim, lo, hi)         # existing
    evaluate_population(pop, evaluate_func)              # existing
    fitness_history = []

    for gen in range(n_generations):
        offspring = custom_select(pop, len(pop))        # the slot above
        offspring = [clone(ind) for ind in offspring]   # existing

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])  # the slot above
                invalidate_fitness(offspring[i], offspring[i + 1])

        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)      # the slot above
                invalidate_fitness(offspring[i])

        for ind in offspring:
            clip_individual(ind, lo, hi)                 # existing bound enforcement

        evaluate_population(offspring, evaluate_func)    # only re-evaluate changed ones
        pop[:] = offspring

        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```

The three `# TODO` operator bodies are the only empty slots; the selection rule, the
recombination rule, and the mutation rule are exactly what the design has to fill in.

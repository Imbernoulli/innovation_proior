# Differential Evolution (DE/rand/1/bin), distilled

Differential Evolution is a population-based direct-search method for continuous black-box global
optimization. Its one idea: generate the perturbation of a vector from the **scaled difference of
two other randomly chosen population members**, so the step's scale and orientation are taken from
the population's own current spread instead of from an externally tuned distribution or schedule.

## Problem it solves

Minimize a scalar cost `f(x)`, `x ∈ R^D`, that is nonlinear, non-differentiable, and multimodal —
no gradient, possibly from a slow simulation or physical experiment — reaching the global minimum
reliably, with **few robust control variables** and easy parallelization across expensive
evaluations.

## Key idea

Maintain a fixed-size population of `NP` real vectors. To improve each member (the *target*
`x_i`), build a candidate (the *trial* `u`) from three steps, then accept greedily:

- **Mutation (DE/rand/1).** Pick three other members `x_{r1}, x_{r2}, x_{r3}`, mutually distinct
  and distinct from `i` (so `NP ≥ 4`), and form
  `v = x_{r1} + F · (x_{r2} − x_{r3})`.
  The difference `x_{r2} − x_{r3}` is a sample of the population's current distribution of pairwise
  differences, so its magnitude is large when the population is spread (early, exploring) and small
  when it has clustered (late, refining) — the step **self-scales** with no schedule. The
  differences also inherit the **orientation** of the population cloud, so along a curved/elongated
  valley they point down the valley — anisotropic search for free, no covariance to estimate.
- **Crossover (binomial).** Mix the mutant with the target per coordinate:
  `u_j = v_j if (rand_j < CR) or j = j_rand, else x_{i,j}`,
  with `j_rand` a randomly chosen coordinate **forced** to come from `v` so the trial is not the
  all-target copy. Unless that forced mutant value happens to equal the target value exactly, this
  avoids wasting an evaluation on an unchanged point. `CR` near 1 ⇒ trial ≈ full difference-vector
  move (good for non-separable problems); `CR` near 0 ⇒ change few coordinates (good for separable
  ones).
- **Selection (greedy, one-to-one).** `u` takes slot `i` in the next generation iff
  `cost(u) ≤ cost(x_i)`, else `x_i` is kept. Per-slot elitism makes the best cost monotone; one-to-one
  replacement (trial vs *its own* target, not global truncation) preserves population diversity,
  which is what lets trapped members be pulled out of local minima by differences from members
  exploring other basins.

Naming: `DE/x/y/z` = base vector (`rand`|`best`) / number of difference vectors / crossover scheme
(`bin`). The workhorse is **DE/rand/1/bin**.

## Control variables and why

Only three, all robust and easy to set:

- **`F`** (scale factor, `0 < F ≤ 2`, typically `≈ 0.5`): a single multiplier on an already correctly
  sized-and-oriented difference vector. Too small ⇒ premature convergence (steps die before reaching
  the optimum); too large ⇒ overshoot, no convergence. Not adapted over time — the difference vector
  already supplies the time-adaptation.
- **`CR`** (crossover rate `∈ [0,1]`, typically `≈ 0.9`): fraction of coordinates taken from the
  mutant; tunes the search to the problem's separability.
- **`NP`** (population size, typically several to `~10·D`, and `≥ 4`): sets the richness of the
  difference-vector set; too small ⇒ stagnation, too large ⇒ wasted evaluations.

Population initialized uniformly over the box `[lo, hi]^D` (or around a known nominal solution plus
small random deviations, if one exists).

## Algorithm

```
initialize population {x_1, ..., x_NP} uniformly over [lo, hi]^D;  evaluate all
require NP >= 4
for G = 1 .. n_generations:
    for i = 1 .. NP:                                  # one competition per target
        pick r1, r2, r3 in {1..NP}, distinct and != i
        v  = x_r1 + F * (x_r2 - x_r3)                 # DE/rand/1 mutation
        j_rand = random coordinate in {1..D}
        for j = 1 .. D:                               # binomial crossover
            u_j = v_j  if (rand() < CR or j == j_rand) else x_{i,j}
        clip u into [lo, hi]
        evaluate f(u)
        if f(u) <= f(x_i):  x_i <- u                  # greedy one-to-one selection
    record best cost in the population
return best individual, history of best cost per generation
```

## Related schemes

- **DE/best/1/bin**: mutation base is the current best, `v = x_best + F·(x_{r1} − x_{r2})` — greedier,
  faster on easy unimodal problems, but less diverse and riskier on multimodal ones.
- **DE/best/2/bin**: two difference vectors off the best,
  `v = x_best + F·(x_{r1} + x_{r2} − x_{r3} − x_{r4})`.
- **Exponential crossover**: take a *contiguous* block of coordinates from the mutant instead of
  choosing coordinates independently. Binomial avoids positional bias and gives `CR` a direct
  "fraction of coordinates from the mutant" reading.

## Working code

Faithful DE/rand/1/bin filling the population-harness generation slot. It follows
`G -> G+1` generation bookkeeping by building each trial from the current population, uses the DEAP-style
fitness container, and fixes the task controls at `F = 0.5`, `CR = 0.9`:

```python
import random
from typing import Tuple, Callable
from deap import base, creator, tools

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def make_individual(toolbox, dim: int, lo: float, hi: float):
    return creator.Individual([random.uniform(lo, hi) for _ in range(dim)])


def clip_individual(individual, lo: float, hi: float):
    for i in range(len(individual)):
        individual[i] = max(lo, min(hi, individual[i]))
    return individual


def make_generation(population: list, toolbox, dim: int, lo: float, hi: float) -> list:
    """One DE/rand/1/bin generation with F = 0.5, CR = 0.9."""
    F = 0.5
    CR = 0.9
    if len(population) < 4:
        raise ValueError("DE/rand/1/bin needs pop_size >= 4")

    next_population = list(population)

    for i, target in enumerate(population):
        # Three other members, mutually distinct and different from target i.
        candidates = list(range(len(population)))
        candidates.remove(i)
        r1, r2, r3 = random.sample(candidates, 3)
        x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]

        # DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3).
        mutant = creator.Individual(
            [x_r1[j] + F * (x_r2[j] - x_r3[j]) for j in range(dim)]
        )

        # Binomial crossover, with one forced coordinate from the mutant.
        j_rand = random.randrange(dim)
        trial = creator.Individual(
            [mutant[j] if (random.random() < CR or j == j_rand) else target[j]
             for j in range(dim)]
        )
        clip_individual(trial, lo, hi)
        trial.fitness.values = toolbox.evaluate(trial)

        # Greedy one-to-one selection: trial replaces target iff its cost is no worse.
        if trial.fitness.values[0] <= target.fitness.values[0]:
            next_population[i] = trial

    return next_population


def run_evolution(evaluate_func: Callable, dim: int, lo: float, hi: float,
                  pop_size: int, n_generations: int,
                  cx_prob: float, mut_prob: float, seed: int) -> Tuple[list, list]:
    """Differential Evolution: DE/rand/1/bin. F = 0.5, CR = 0.9."""
    random.seed(seed)
    _ = (cx_prob, mut_prob)  # retained for harness compatibility; F and CR are fixed above

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)            # init uniformly over the box
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitness_history = []

    for gen in range(n_generations):
        pop = make_generation(pop, toolbox, dim, lo, hi)

        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```

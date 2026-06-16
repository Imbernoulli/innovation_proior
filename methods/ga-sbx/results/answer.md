# Simulated Binary Crossover (SBX) — real-coded GA, distilled

Simulated Binary Crossover is a real-parameter recombination operator that reproduces, on
continuous variables directly, the offspring-distribution behaviour of single-point
crossover on binary strings — without ever binary-encoding the reals. Paired with
polynomial mutation and tournament selection, it gives a genetic algorithm that searches
continuous black-box objectives without the precision caps, fixed mappings, and
Hamming-cliff pathologies of a binary-coded GA.

## Problem it solves

Minimize a black-box `f: [lo, hi]^n -> R` with a genetic algorithm operating on
real-valued vectors. Binary-coded GAs search discrete spaces well (single-point crossover
propagates building blocks), but encoding reals in bits caps precision, fixes the
mapping, and creates Hamming cliffs (value-adjacent reals are bit-distant). SBX is a
crossover that acts on the reals directly while inheriting what made single-point
crossover effective.

## Key idea

Single-point binary crossover, read on the *decoded* reals, has two properties: it
preserves the parents' mean (children are equidistant from the parents' midpoint), and
its **spread factor** `beta = |c1 - c2| / |p1 - p2|` is distributed with a strong peak
near `beta = 1` (children near the parents far more likely than children far away). SBX
reproduces both directly in real space:

- Choose a **spread-factor density** with a single distribution index `eta_c >= 0`:

  ```
  C(beta) = 0.5 (eta_c + 1) beta^eta_c            for 0 <= beta <= 1   (contracting)
  E(beta) = 0.5 (eta_c + 1) / beta^(eta_c + 2)    for beta > 1         (expanding)
  ```

  The expanding side is forced by the symmetry `E(beta) = beta^-2 C(1/beta)` (crossing a
  contracting crossover's children returns the parents, pairing `beta` with `1/beta`).
  Each half carries probability 0.5. Large `eta_c` concentrates children near the
  parents (exploitation); `eta_c = 0` makes the contracting half uniform, matching
  the inside-the-parents part of BLX-0.0 while keeping the expanding tail.

- **Sample `beta` by inverse-transform** (closed-form, two ops, no special functions).
  With `u ~ U[0,1]`:

  ```
  beta = (2u)^(1/(eta_c+1))                if u <= 0.5
  beta = (1 / (2(1-u)))^(1/(eta_c+1))      if u >  0.5
  ```

  These invert the CDFs `0.5 beta^(eta_c+1)` (contracting) and
  `1 - 0.5 beta^-(eta_c+1)` (expanding).

- **Build offspring** from `beta` so that the mean is preserved and the spread is
  honored:

  ```
  c1 = 0.5 [ (1 + beta) x1 + (1 - beta) x2 ]
  c2 = 0.5 [ (1 - beta) x1 + (1 + beta) x2 ]
  ```

  Then `c1 + c2 = x1 + x2` (mean preserved) and `c1 - c2 = beta (x1 - x2)` (spread =
  `beta`). In the `ga_sbx` baseline, DEAP's unbounded `cxSimulatedBinary` applies this
  kernel to every coordinate once the outer crossover probability `p_c` gates a mating.

Because `beta` is a *ratio*, the spread scales with the parents' separation: a converging
population (close parents) automatically narrows its search — self-annealing exploration.

## Polynomial mutation

A local real-coded perturbation with the same polynomial shape, peaked at the parent,
controlled by index `eta_m`, that can never leave `[lo, hi]`. For value `x`, with
`delta_1 = (x - lo)/(hi - lo)`, `delta_2 = (hi - x)/(hi - lo)`, `u ~ U[0,1]`,
`mut_pow = 1/(eta_m + 1)`:

```
u < 0.5 :  delta_q = [ 2u + (1 - 2u)(1 - delta_1)^(eta_m+1) ]^mut_pow - 1
u >= 0.5:  delta_q = 1 - [ 2(1-u) + 2(u - 0.5)(1 - delta_2)^(eta_m+1) ]^mut_pow
x' = x + delta_q (hi - lo)
```

The boundary terms cap the move: at `lo`, the downward branch gives `delta_q = 0`; at
`hi`, the upward branch gives `delta_q = 0`; as `u -> 0` the downward branch reaches
`-delta_1`, and as `u -> 1` the upward branch reaches `+delta_2`. The unbounded
polynomial branch is recovered only when the chosen side has the full normalized range
available. Perturbation magnitude is `O((hi - lo)/eta_m)`. DEAP applies this per
coordinate with independent probability `indpb = 1/n` when the mutation kernel is
called (the real-coded analogue of the binary `1/L`, about one mutated coordinate per
mutation pass).

## Selection

Tournament selection of size `t`: take `t` individuals uniformly at random, keep the
best under the fitness object's ordering, repeat. No fitness scaling or sorting; in DEAP,
minimization is handled by negative fitness weights, not by changing the selector.
`t` is the selection-pressure knob. Default `t = 3` (mild pressure).

## Defaults and why

- `eta_c = eta_m = 20`: focused enough that children/mutants hug good sources
  (exploitation) while a thinning tail still allows occasional large jumps; `eta_c ~ 2`
  is the broad single-objective choice, `~20` the common balanced default.
- `indpb = 1/n`: about one coordinate is mutated per mutation pass — keeps diversity
  without tearing good solutions apart.
- DEAP unbounded SBX on every coordinate of a selected mating pair; polynomial mutation
  uses `mutPolynomialBounded(..., indpb=1/n)`; tournament size 3.

## Final algorithm

```
initialize population of float vectors uniformly in [lo, hi]^n; evaluate
for each generation:
    parents   <- tournament-select(pop, |pop|, t=3)
    offspring <- clone(parents)
    for each pair (a, b) in offspring:  with prob p_c:  SBX(a, b, eta_c=20)
    for each ind in offspring:          with prob mut_prob:  poly_mutate(ind, eta_m=20, indpb=1/n)
    clip offspring into [lo, hi]
    evaluate changed offspring
    pop <- offspring                                   # generational replacement
return best individual, history of best fitness per generation
```

## Working code

Faithful to the DEAP `tools` operators (`cxSimulatedBinary`, `mutPolynomialBounded`,
`selTournament`), filling the three operator slots of the generational GA harness.

```python
import random
import numpy as np
from typing import Callable, Tuple
from deap import base, tools


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Tournament selection, tournament size 3; Fitness weights define best."""
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Simulated Binary Crossover, distribution index eta_c = 20."""
    eta = 20.0
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        u = random.random()
        if u <= 0.5:                              # contracting: beta = (2u)^(1/(eta+1))
            beta = 2.0 * u
        else:                                     # expanding: beta = (1/(2(1-u)))^(1/(eta+1))
            beta = 1.0 / (2.0 * (1.0 - u))
        beta **= 1.0 / (eta + 1.0)
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))   # mean-preserving,
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))   # spread = beta
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Polynomial bounded mutation, index eta_m = 20, per-variable rate 1/n."""
    eta = 20.0
    indpb = 1.0 / len(individual)
    for i in range(len(individual)):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - lo) / (hi - lo)         # normalized room below x
            delta_2 = (hi - x) / (hi - lo)         # normalized room above x
            u = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if u < 0.5:                            # downward, capped at -delta_1
                xy = 1.0 - delta_1
                val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)
                delta_q = val ** mut_pow - 1.0
            else:                                  # upward, capped at +delta_2
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)
                delta_q = 1.0 - val ** mut_pow
            x = x + delta_q * (hi - lo)
            individual[i] = min(max(x, lo), hi)    # numerical safety clip
    return (individual,)


def run_evolution(
    evaluate_func: Callable, dim: int, lo: float, hi: float,
    pop_size: int, n_generations: int, cx_prob: float, mut_prob: float, seed: int,
) -> Tuple[list, list]:
    """Generational real-coded GA: tournament + SBX + polynomial mutation."""
    random.seed(seed)
    np.random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit

    fitness_history = []
    for gen in range(n_generations):
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        for ind in offspring:
            clip_individual(ind, lo, hi)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        fitness_history.append(min(ind.fitness.values[0] for ind in pop))

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```

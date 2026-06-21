# Context: evolutionary multi-objective optimization (circa 1999-2001)

## Research question

A multi-objective optimization problem asks for a decision vector `x ∈ X` that minimizes a
vector of `m` conflicting objectives `f(x) = (f_1(x), …, f_m(x))`. Because the objectives
conflict, there is no single best `x`; instead there is a *set* — the Pareto-optimal set, all
`x` for which no other decision vector improves one objective without worsening another. An
evolutionary algorithm is run once and must return a good finite *approximation* of that whole
set. The returned set should be close to the true front (convergence) and well spread along it
with the extremes retained (diversity). The question is how to design the two selection
mechanisms a population-based search needs — *mating selection* (which individuals reproduce)
and *environmental selection* (which individuals survive into the next generation and into a
bounded elite archive) — so that the surviving set is simultaneously converged, uniformly
spread, and inclusive of the boundary trade-offs, on problems with two and three or more
objectives.

## Background

After the first studies on evolutionary multi-objective optimization (EMO) in the mid-1980s, a
wave of Pareto-based techniques appeared in 1993-94 — MOGA (Fonseca and Fleming 1993), NPGA
(Horn, Nafpliotis, and Goldberg 1994), and NSGA (Srinivas and Deb 1994) — demonstrating that a
single evolutionary run can approximate the trade-off surface. These early methods did not use
*elitism* explicitly. A few years later, experimental work (Parks and Miller 1998; Zitzler,
Deb, and Thiele 2000) recognized and supported the importance of elitism — retaining the best
solutions found so far in an external archive — and a cluster of elitist EMO algorithms
followed (SPEA, PAES, and then PESA and NSGA-II).

The load-bearing concepts:

- **Pareto dominance.** For a minimization problem, `a` dominates `b` (`a ≻ b`) iff `a` is no
  worse in every objective and strictly better in at least one. Dominance is a partial order:
  two individuals that each win on a different objective are *indifferent*. The nondominated
  individuals within a set form its current trade-off front.

- **Mating vs. environmental selection.** Modern EMO splits selection in two. The pool at each
  generation is ranked (mating selection, usually randomized via tournaments) and, separately,
  pruned to keep only some fraction for the next generation (environmental selection, usually
  deterministic). The two can in principle use different criteria.

- **External archive / elitism.** An archive holds a representation of the best nondominated
  front seen so far. A member is removed only if a solution dominating it is found or if the
  archive overflows and its region is overcrowded. Being copied into the archive is the only
  way an individual survives many generations rather than by chance.

- **Density / niching.** When dominance cannot discriminate, a secondary signal can measure how
  *crowded* the neighborhood of an individual is, so that selection can prefer individuals in
  sparse regions and spread the population over the front. Two families exist by this time: a
  per-objective, sort-based crowding measure (used in NSGA-II), and a hyper-grid / histogram
  count of how many individuals share an individual's cell (used in PESA). A third, older idea
  from statistics is the **k-th nearest neighbor density estimator** (Silverman 1986): the
  density at a point is a decreasing function of the distance to its `k`-th nearest data point;
  it adapts the bandwidth to the local density and its standard smoothing choice is `k` equal to
  about the square root of the sample size.

## Baselines

The prior elitist EMO methods a new algorithm would be measured against and would react to.

**SPEA — Strength Pareto Evolutionary Algorithm (Zitzler and Thiele, IEEE TEC 1999).** Keeps a
regular population `P` and an external archive `P̄`. Each generation: copy nondominated members
into `P̄`, drop dominated members and duplicates, and if `P̄` overflows a preset limit, reduce it
by **clustering**. Fitness rests on a *strength* value: each archive member `i` is given a
strength `S(i) ∈ [0,1)` equal to the number of population members it dominates (or equals)
divided by `N + 1`, and this `S(i)` is also its fitness. A population member `j` then gets
fitness `F(j) = 1 + Σ_{i ∈ P̄, i ≽ j} S(i)` — one plus the strengths of the archive members that
cover it. Fitness is minimized, so archive members (small `S`) are always fitter than population
members, giving elitism.

**NSGA-II (Deb, Agrawal, Pratap, and Meyarivan 2000).** Sorts the combined
parent+offspring pool into nondomination fronts `F_1, F_2, …` (front index = rank), then within
a front uses a **crowding distance**: for each objective, sort the front and give each interior
member the normalized gap between its two neighbors, summed over objectives, with the two
boundary members of each objective set to infinity so they are always kept. The
crowded-comparison operator prefers a lower rank, and within equal rank prefers the larger
crowding distance (the less crowded individual). The next generation is filled front by front,
the last partial front sorted by crowding distance. Complexity `O(m N^2)`.

**PESA — Pareto Envelope-based Selection Algorithm (Corne, Knowles, and Oates 2000).** Mating
selection acts only on an archive that stores (a subset of) the current nondominated set. Its
density measure is a **histogram / hyper-grid** technique: the objective space is divided into
cells and members are sampled inversely to how crowded their cell is. Offspring are checked for
inclusion in the archive after each generational cycle.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Continuous test functions**, coded as real vectors with 100 decision variables each, with
  recombination by the SBX operator (distribution index 20; Deb and Agrawal 1995) and a
  polynomial mutation distribution: the multi-objective Sphere model SPH-`m` (two- and
  three-objective instances, large domain `[−10^3, 10^3]^n`); ZDT6 (Zitzler, Deb, and Thiele
  2000), unimodal with a non-uniformly distributed objective space; QV (Quagliarella and Vicini
  1997); and KUR (Kursawe 1991), on a large domain. Population and archive sizes both 100.
- **Combinatorial test functions**: the multi-objective 0/1 knapsack KP-750-`m` (Zitzler and
  Thiele 1999) with 750 items and `m = 2, 3, 4` knapsacks, individuals as bit strings, one-point
  crossover, bit-flip mutation at probability 0.006, with a greedy repair for capacity
  constraints; population/archive 250 for `m=2`, 300 for `m=3`, 400 for `m=4`.
- Protocol: each algorithm uses identical population and archive sizes and identical
  recombination/mutation/sampling operators, differing only in the fitness assignment and
  selection; multiple independent runs per problem to average out random effects.
- Quality is judged on convergence (closeness to the true front), diversity, and spread of the
  returned nondominated set; the per-objective benchmark domains and operators above are the
  fixed setting.

## Code framework

The method plugs into a generic multi-objective evolutionary loop that already exists. The data
representation (real or bit vectors), the variation operators (a bounded simulated-binary
crossover and a bounded polynomial mutation), and the dominance relation on fitness tuples are
all standard library pieces. The open design questions are how to turn the dominance partial
order into a usable scalar ranking, how to pick parents, and how to prune the combined pool
back to a fixed size.

```python
from copy import deepcopy
import random

from deap import tools     # SBX crossover, polynomial mutation, dominance on fitness


class CustomMOEA:
    """A generic multi-objective evolutionary strategy: it owns the problem
    parameters and the three selection/variation hooks an EMO loop calls each
    generation. Individuals expose ind.fitness.values (a tuple, all minimized),
    ind.fitness.dominates(other.fitness), and ind.fitness.valid."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                 # (low, up) per gene
        self.cx_eta = cx_eta                 # SBX distribution index
        self.mut_eta = mut_eta               # polynomial-mutation distribution index
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        """Mating selection: choose k parents for reproduction.
        # TODO: the parent-selection rule we will design."""
        pass

    def vary(self, parents):
        """Produce offspring by recombination + mutation (invalidate fitness)."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:                      # crossover probability
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        """Environmental selection: from the combined parent+offspring pool,
        keep exactly pop_size individuals for the next generation.
        # TODO: the survival rule we will design — the scalar fitness that ranks
        #       even mutually-nondominated individuals, and the fixed-size
        #       reduction of the combined pool."""
        combined = population + offspring
        pass

    def on_generation(self, gen, population):
        """Optional per-generation hook for adaptive strategies."""
        pass


# the EMO driver that already exists and calls the hooks above
def evolve(strategy, population, evaluate, n_gen):
    for ind in population:
        ind.fitness.values = evaluate(ind)
    for gen in range(n_gen):
        parents = strategy.select(population, strategy.pop_size)
        offspring = strategy.vary(parents)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)         # objective evaluation
        population = strategy.survive(population, offspring)
        strategy.on_generation(gen, population)
    return population
```

The `vary` hook is fixed library machinery (bounded SBX + polynomial mutation); the work is
in `select` and `survive` — the scalar fitness, the parent selection, and the fixed-size
reduction.

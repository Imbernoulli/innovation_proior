## Research question

A problem with several conflicting objectives `f(x) = (f_1(x), ..., f_M(x))`, all to be minimized, has no
single best solution: improving one objective forces another to get worse, so the answer is a *set* — the
Pareto-optimal points, those that no feasible point beats on every objective at once. The job is to recover
a good approximation of that whole front in **one** run. That splits into two pressures that fight each
other: **convergence** (drive the population down onto the true front) and **diversity/spread** (keep the
survivors spaced across the front and reaching its extremes). A method that only converges collapses to a
crowded blob; one that only spreads never lands. The single thing being designed here is the **generation
strategy** of a population-based evolutionary algorithm — how parents are picked, how offspring are made,
and (the load-bearing choice) how the combined parent+offspring pool is pruned back to the population size.
Everything else about the harness is fixed.

## Prior art before the first rung

The first rung reacts to the late-1990s elitist-MOEA template. These are the ancestors the ladder climbs
away from; each buys one or two of {cheap, elitist, parameter-free diversity} and pays elsewhere.

- **Layered ranking + fitness sharing (Goldberg 1989; Goldberg & Richardson 1987; Srinivas & Deb 1994).**
  Rank the population into Pareto layers, then keep each layer spread with a sharing kernel of radius
  `sigma_share`. Gap: the ranking is `O(M N^3)` as usually implemented, it is *non-elitist* (parents can be
  lost between generations), and the spread depends on the fragile `sigma_share`.
- **SPEA — Strength-Pareto EA (Zitzler & Thiele 1998/99).** Keeps an *external archive* of non-dominated
  solutions that participates in selection; fitness is a strength count, diversity is by clustering the
  archive when it overflows. Gap: couples archive sizing, strength fitness, and a separate clustering step
  — heavier than a plain population loop.
- **Tchebycheff scalarization (Bowman 1976; Miettinen 1999).** Turn the vector objective into a scalar
  `max_j w_j |f_j - z*_j|` against an ideal point `z*` and a weight `w`. A single weight finds one front
  point; sweeping weights traces the front, but choosing and managing the weights — and which subproblems
  cooperate — is left open. This is the lineage the first rung (decomposition) is built on.
- **Real-coded variation (Deb & Agrawal 1995).** Taken as given: simulated binary crossover (SBX, spread
  factor `beta = |(c2-c1)/(p2-p1)|`, distribution index `eta_c`) and bounded polynomial mutation
  (distribution index `eta_m`, per-variable rate `p_m = 1/n`). Both are local-by-default and self-adapt as
  the population contracts; every rung on this ladder reuses them unchanged.

## The fixed substrate

A standard real-coded generational MOEA loop is frozen and must not be touched. It builds a random initial
population inside the box bounds, evaluates it, and then each generation calls — in order — `select` to pick
parents, `vary` to make offspring, evaluates any offspring whose fitness was invalidated, calls `survive` to
prune the combined pool back to `pop_size`, and calls the optional `on_generation` hook. Hypervolume (HV,
higher better), inverted generational distance (IGD, lower better), and spread (lower better) are computed
against a reference Pareto front by the harness; the algorithm never sees them. The loop also hands the
strategy these helpers: `tools.sortNondominated`, `tools.selTournamentDCD`, `tools.selNSGA3`,
`tools.selSPEA2`, `tools.cxSimulatedBinaryBounded`, `tools.mutPolynomialBounded`,
`tools.uniform_reference_points`, plus `compute_crowding_distance(front)` (sets `.fitness.crowding_dist`)
and `get_nondominated(pop)` (the first front). Individuals expose `ind.fitness.values` (the objective
tuple, all minimized), `ind.fitness.dominates(other.fitness)`, and `ind.fitness.valid`.

## The editable interface

Exactly one region is editable — the `CustomMOEA` class in `deap/custom_moea.py` (lines 297–441). Every
method on the ladder is a fill of this same contract: `__init__(pop_size, n_obj, n_var, bounds, cx_eta,
mut_eta, mut_prob)`, `select(population, k)` (pick `k` parents), `vary(parents)` (crossover + mutation,
fitness invalidated), `survive(population, offspring)` (environmental selection back to `pop_size`), and an
optional `on_generation(gen, population)` (adaptive callback). The algorithm must work for both 2-objective
and 3-objective problems.

The starting point is the scaffold default, which is already NSGA-II-shaped: binary tournament on
rank+crowding for `select`, SBX(0.9)+polynomial mutation for `vary`, and non-dominated-sort + crowding for
`survive`. Each method on the ladder replaces exactly these definitions.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — default fill
class CustomMOEA:
    """Custom multi-objective evolutionary algorithm (default: NSGA-II-shaped)."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        # binary tournament on non-domination rank + crowding distance
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(offspring[i], offspring[i + 1],
                                               eta=self.cx_eta, low=lo, up=hi)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        # combine, non-dominated-sort, fill fronts, truncate last front by crowding
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
                next_gen.extend(front[:remaining])
                break
        return next_gen

    def on_generation(self, gen, population):
        pass
```

## Evaluation settings

Four benchmark problems spanning 2- and 3-objective fronts of different shapes, each over three seeds
{42, 123, 456}: **ZDT1** (2-obj, convex front, 30 vars, 200 generations), **ZDT3** (2-obj, disconnected
front, 30 vars, 200 generations), **DTLZ2** (3-obj, spherical front, 12 vars, 250 generations), and
**DTLZ1** (3-obj, linear front riddled with local fronts, 7 vars, 400 generations). Population size is 100
for the ZDTs and 120 for the DTLZs. Three metrics: **HV** (volume of objective space dominated by the
approximation; higher better), **IGD** (mean distance from true-front points to the nearest found solution;
lower better), and **Spread** (uniformity of the approximation; lower better). The reference points are
`[1.1, 1.1]` for the ZDTs, `[1.5, 1.5, 1.5]` for DTLZ2, and `[1.0, 1.0, 1.0]` for DTLZ1.

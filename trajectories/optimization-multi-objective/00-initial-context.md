## Research question

A multi-objective problem `f(x) = (f_1(x), ..., f_M(x))`, all minimized, has no single optimum. The goal is a set of Pareto-optimal points approximated in one run. Two pressures oppose each other: **convergence** toward the true Pareto front and **diversity/spread** across it, including the extremes. The design object is the **generation strategy** of a real-coded population-based MOEA: parent selection, offspring creation, and environmental selection that trims the combined parent+offspring pool back to `pop_size`.

## Prior art / Background / Baselines

- **Layered Pareto ranking + fitness sharing (Srinivas & Deb 1994).** Sort the population into non-domination layers and preserve spread with a sharing kernel of radius `sigma_share`. Gap: the ranking step is expensive, the method is non-elitist, and spread depends on the fragile user-set `sigma_share`.
- **SPEA — Strength-Pareto EA (Zitzler & Thiele 1998/99).** Maintain an external archive of non-dominated solutions; fitness combines a strength count and a raw domination count, and archive overflow is reduced by clustering. Gap: archive sizing, strength assignment, and clustering add overhead and extra parameters.
- **Tchebycheff scalarization (Bowman 1976; Miettinen 1999).** Decompose the front into scalar subproblems `max_j w_j |f_j - z*_j|` with an ideal point `z*` and a weight vector `w`. Gap: choosing, distributing, and adapting the weight vectors, and deciding how subproblems interact, is still open.
- **Real-coded variation (Deb & Agrawal 1995).** Simulated binary crossover (SBX, `eta_c`) and bounded polynomial mutation (`eta_m`, per-variable rate `p_m = 1/n`) are taken as given. Gap: they provide local variation as the population contracts, but no Pareto ranking or elitism by themselves.

## Fixed substrate / Code framework

A standard generational real-coded MOEA loop is fixed. It initializes a random box-bounded population, evaluates it, and each generation calls `select`, `vary`, `survive`, and optionally `on_generation`. The harness computes HV, IGD, and Spread against a reference front; the algorithm does not see them. Helpers provided: `tools.sortNondominated`, `tools.selTournamentDCD`, `tools.selNSGA3`, `tools.selSPEA2`, `tools.cxSimulatedBinaryBounded`, `tools.mutPolynomialBounded`, `tools.uniform_reference_points`, `compute_crowding_distance(front)`, `get_nondominated(pop)`. Individuals expose `fitness.values`, `fitness.dominates(...)`, and `fitness.valid`.

## Editable interface

Only the `CustomMOEA` class in `deap/custom_moea.py` (lines 297–441) is editable. Its contract is `__init__(pop_size, n_obj, n_var, bounds, cx_eta, mut_eta, mut_prob)`, `select(population, k)`, `vary(parents)`, `survive(population, offspring)`, and optional `on_generation(gen, population)`. The algorithm must work for both 2- and 3-objective problems.

The starting point is the default scaffold below.

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

Four benchmarks over seeds {42, 123, 456}, spanning 2- and 3-objective fronts: **ZDT1** (convex, 30 vars, 200 gens), **ZDT3** (disconnected, 30 vars, 200 gens), **DTLZ2** (spherical, 12 vars, 250 gens), and **DTLZ1** (linear with local fronts, 7 vars, 400 gens). Population size is 100 for ZDT and 120 for DTLZ. Metrics: **HV** (higher better), **IGD** (lower better), and **Spread** (lower better). Reference points are `[1.1, 1.1]` for ZDT, `[1.5, 1.5, 1.5]` for DTLZ2, and `[1.0, 1.0, 1.0]` for DTLZ1.

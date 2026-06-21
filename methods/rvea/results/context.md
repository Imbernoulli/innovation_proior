# Context: approximating the Pareto front when the number of objectives is large

## Research question

A multi-objective optimization problem (MOP) asks to minimize `M` conflicting objectives at
once, `F(x) = (f_1(x), ..., f_M(x))` over a decision space `X ⊆ R^n`. Because the objectives
conflict, no single `x` minimizes all of them; the meaningful answer is the *Pareto set* — the
points no other feasible point is at least as good on every objective and strictly better on
one (the non-dominated points) — and its image in objective space, the *Pareto front* (PF). A
decision maker does not want the whole, often infinite, PF; they want a *finite, evenly
spread, well-converged* set of representative trade-offs from a single run, both **close to**
the true PF (convergence) and **well distributed along it** (diversity).

The specific regime of interest is *many*-objective optimization: `M` larger than three. How
should an evolutionary algorithm approximate the Pareto front in this regime, where a fixed
population budget must be distributed across a high-dimensional objective space and the
objectives may differ by orders of magnitude in scale?

## Background

By the early-to-mid 2010s evolutionary algorithms (EAs) are the dominant tool for
approximating the PF, because a population naturally carries many candidate trade-offs at once
and the objectives need not be differentiable or convex. The field state and the load-bearing
concepts:

- **Pareto dominance and the partial order.** `x` dominates `y` iff `f_i(x) <= f_i(y)` for
  all `i` and strictly for at least one. Dominance is the bedrock comparison in multi-objective
  EAs, but it is only a *partial* order: two points can be mutually non-dominated and hence
  incomparable. As the number of objectives grows, the proportion of mutually non-dominated
  candidate solutions in a finite population rises toward one even at an early stage of the
  search, so dominance can separate fewer and fewer individuals.

- **Diversity in high-dimensional objective space is sparse.** Since the PF of most continuous
  MOPs is a piecewise-continuous manifold, an EA with a fixed population can only return a
  representative *sample* of it. For `M = 2` or `3` the PF is a curve or a surface and keeping
  the sample evenly spread is relatively easy. As `M` grows the candidate solutions sit very
  sparsely in the objective space and density-based diversity machinery widely used in MOEAs
  (e.g. crowding distance in NSGA-II) has finer resolution requirements.

- **Decomposition and scalarization — the classical theory.** A cornerstone of traditional
  (non-evolutionary) multi-objective optimization: a complex MOP can be divided into a number
  of simpler sub-problems and solved collaboratively. Two flavours are relevant. In the first,
  the MOP is decomposed into a group of single-objective problems through a *scalarization*
  attached to a weight vector `w = (w_1, ..., w_M)`, `w_i >= 0`, `sum_i w_i = 1`. The standard
  aggregation functions, with their classical properties:
  - **Weighted sum:** `g^ws(x | w) = sum_i w_i f_i(x)`. Simple; minimizing a weighted sum
    reaches only the supported Pareto points exposed by supporting hyperplanes of the
    attainable objective set.
  - **Tchebycheff:** `g^te(x | w, z*) = max_i { w_i |f_i(x) - z*_i| }`, with the ideal point
    `z*_i = min_{x in X} f_i(x)`. Sweeping `w` can in principle trace the whole PF, convex or
    not.
  - **Penalty-based boundary intersection (PBI; Zhang & Li 2007).** With the normalized
    direction `u = w / ||w||`, decompose the displacement of an objective vector from the
    reference point into a component *along* the line and a component *off* it:
    `d_1 = ||(F(x) - z*)^T u||` (distance along the line through `z*` in direction `w`) and
    `d_2 = ||(F(x) - z*) - d_1 u||` (perpendicular distance off that line). The scalar
    objective is `g^pbi(x | w, z*) = d_1 + theta · d_2`, with `theta > 0` a fixed penalty
    (commonly `theta = 5`). `d_1` is a convergence term, `d_2 · theta` a diversity term that
    keeps solutions near the prescribed line.
  In the second flavour, the MOP is decomposed into a group of *sub-MOPs* by partitioning the
  objective space into subspaces (MOEA/D-M2M, Liu et al. 2014, divides the PF into segments by
  direction vectors; IM-MOEA, Cheng et al. 2015, partitions the objective space with reference
  vectors). Different works call the partitioning vectors *direction vectors*, *reference
  lines*, or *reference vectors*; they all play the role of carving the objective space into
  subspaces inside which selection happens.

- **Performance-indicator selection (IBEA, SMS-EMOA, HypE).** A separate family that ranks
  solutions by the contribution to a set-quality indicator, most often the hypervolume. The
  indicator computation (the hypervolume in particular) grows in cost as `M` grows.

- **Evenly spread reference directions (Das & Dennis simplex-lattice).** The standard way to
  place vectors uniformly: take every vector `u_i` whose components come from
  `{0/H, 1/H, ..., H/H}` and sum to 1; this yields `N = C(H + M - 1, M - 1)` points on the
  unit hyperplane, evenly distributed. For `M >= 8` a two-layer construction (an outer layer
  with `H_1` divisions plus a smaller inner layer with `H_2`) keeps the count manageable. These
  hyperplane points can be mapped onto the unit hypersphere by dividing each by its norm,
  `u_i / ||u_i||`, putting unit-length direction vectors in the first quadrant.

- **The angle between two vectors.** For vectors `v_1, v_2`, the cosine of the acute angle is
  `cos θ = (v_1 · v_2) / (||v_1|| ||v_2||)`. When both vectors are unit-length the denominator
  is one. An angle lies in `[0, π/2]` for first-quadrant vectors — a bounded, dimensionless
  quantity, unlike a Euclidean distance, which is unbounded and scales with the magnitude of
  the objective values.

- **The ideal point and objective translation.** `z*_i = min_{x in X} f_i(x)` is unknown
  during the search (computing it exactly would mean solving `M` single-objective problems), so
  the per-objective minimum of the evaluated selection population is the natural cheap
  substitute at a generation. Subtracting it from every objective vector — a rigid translation
  — moves the ideal point to the origin and puts the translated vectors in the first quadrant,
  after which the length of a translated objective vector *is* its distance to the (substitute)
  ideal point.

- **Real-coded variation operators (the NSGA-II C-code lineage).** For continuous box-bounded
  decision variables, the standard recombination/mutation pair: **simulated binary crossover
  (SBX; Deb & Agrawal 1995)**, which spreads two children around the parents with a spread
  factor sampled from a polynomial distribution governed by a distribution index `eta_c` (large
  `eta_c` keeps children near the parents), and **polynomial mutation (Deb & Goyal)**, which
  perturbs each variable with probability `p_m` by an amount drawn from a polynomial
  distribution with index `eta_m`. Both have bounded forms respecting the box `[x_l, x_u]`.

- **The μ+λ elitism shell.** The NSGA-II template: breed an offspring population, combine
  parents and offspring into a pool of size `2N`, then select `N` survivors from the combined
  pool. This generational elitism shell is reusable independently of *how* the `N` survivors
  are chosen.

## Baselines

These are the prior methods a new many-objective EA would be measured against and would react
to.

**NSGA-II (Deb, Pratap, Agarwal & Meyarivan, *IEEE TEC* 6(2), 2002).** The most successful
multi-objective EA of its era. From parents `P_t` of size `N` it breeds offspring `Q_t`
(tournament selection + SBX + polynomial mutation), combines `R_t = P_t ∪ Q_t` (size `2N`),
applies fast non-dominated sorting into fronts `F_1, F_2, ...`, and fills the next population
front by front; the last front that does not fit is truncated by **crowding distance** (per
objective, sort the front, give extremes infinite distance, give each interior point the summed
normalized neighbour gap), keeping the most isolated.

**SPEA2 (Zitzler, Laumanns & Thiele, EUROGEN 2001 / TIK-Report 103).** Strength-based fitness
plus a `k`-nearest-neighbour density estimate (`k = sqrt(N + |archive|)`) with an external
archive.

**MOEA/D-PBI (Zhang & Li, *IEEE TEC* 11(6), 2007).** Decomposes the MOP into `N` scalar
sub-problems on evenly spread weight vectors and optimizes them cooperatively through a
neighbourhood structure; with the PBI aggregation `g^pbi = d_1 + theta · d_2` each sub-problem
balances a convergence term `d_1` against a diversity term `theta · d_2`.

**NSGA-III (Deb & Jain, *IEEE TEC* 18(4), 2014).** Reference-point niching layered on
dominance for many-objective problems. Each generation it estimates the ideal point and the
extreme points, builds the intercepts, and **normalizes the objectives** onto the unit
hyperplane; it then associates each solution with its nearest reference line by perpendicular
distance and niches to preserve diversity, with non-dominated sorting still doing the
convergence work.

**MOEA/D-M2M (Liu, Gu & Zhang, *IEEE TEC* 18(3), 2014) and IM-MOEA (Cheng, Jin & Narukawa,
2015).** The decomposition-into-subspaces line: M2M partitions the PF into segments by
direction vectors, each segment a sub-MOP solved within its own subspace; IM-MOEA partitions
the objective space with reference vectors and builds inverse models inside each subspace.

## Evaluation settings

The natural yardsticks already in use:

- **Continuous bi-objective ZDT suite (Zitzler, Deb & Thiele 2000):** ZDT1 (convex front),
  ZDT3 (disconnected front), with `n = 30` decision variables in `[0,1]^n`.
- **Continuous many-objective DTLZ suite (Deb, Thiele, Laumanns & Zitzler 2002):** DTLZ1
  (linear front with many local fronts), DTLZ2 (spherical/concave front), DTLZ3, DTLZ4, with
  objective numbers `M ∈ {3, 6, 8, 10}` and `n = M + K - 1` decision variables (`K = 5` for
  DTLZ1, `K = 10` for DTLZ2–4). Scaled variants SDTLZ1/SDTLZ3 multiply objective `i` by
  `p^{i-1}` to create badly-scaled fronts.
- **WFG test suite (Huband et al. 2006):** WFG1–WFG9, introducing nonseparability, deception,
  bias in the decision space and mixed PF geometries, `n = K + L - 1` with `L = 10`.
- **Metrics:** *Hypervolume (HV)* — the volume of objective space dominated by the
  approximation and dominating a reference point (higher is better); *Inverted Generational
  Distance (IGD)* — the average distance from a dense set of true-PF points to the nearest
  found solution (lower is better, captures convergence and spread together); and *Spread* /
  uniformity (lower is better). Protocol: multiple independent seeds; identical population size
  and number of function evaluations across compared algorithms; SBX with `eta_c` and
  polynomial mutation with `eta_m`, `p_c = 1.0`, `p_m = 1/n`. Population size set from the
  simplex-lattice factor `H` and `M`. Termination by a maximal number of generations.

## Code framework

A new method plugs into the same generation-based multi-objective EA harness already used for
the baselines. The MOP, the box-bounded real decision space, and the standard real-coded
operators (SBX crossover, polynomial mutation) already exist as library primitives, as do
baseline selection/sorting utilities and the Das-Dennis reference-point generator. What is
*not* settled is how the algorithm chooses who mates, how it produces offspring, and — the
heart of it — *how it decides which individuals survive* from one generation to the next, and
whether it keeps any per-generation adaptive state. Those are the slots to be designed. The
substrate is therefore the generic per-generation loop plus the empty hooks.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # existing utilities: bounded SBX, polynomial mutation,
                         # uniform_reference_points (Das-Dennis simplex-lattice), baseline sorting


class CustomMOEA:
    """Generic multi-objective EA. Owns problem parameters and the standard
    real-coded operators; the selection / variation / survival rules and any
    adaptive state are the empty slots to be designed. Each individual exposes
    ind.fitness.values (a tuple of objective values, all minimized),
    ind.fitness.dominates(other), and ind.fitness.valid."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index
        self.mut_eta = mut_eta                     # polynomial-mutation distribution index
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        # TODO: any state the algorithm we design needs (set up here).

    def select(self, population, k):
        """Choose k parents from the population for mating."""
        # TODO: the parent-selection rule we will design.
        pass

    def vary(self, parents):
        """Produce offspring from the parents via crossover + mutation
        (offspring fitness invalidated so the harness re-evaluates)."""
        # TODO: the variation rule we will design.
        pass

    def survive(self, population, offspring):
        """Environmental selection: return the pop_size individuals that carry
        into the next generation, chosen from the combined parent+offspring pool."""
        # TODO: the survival rule we will design.
        pass

    def on_generation(self, gen, population):
        """Optional per-generation callback (e.g. for adaptive state)."""
        # TODO: any per-generation adaptation we will design.
        pass


# existing generation-based driver the algorithm plugs into
def run(algo, toolbox, n_gen):
    pop = toolbox.init_population(algo.pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)     # evaluate the M objectives
    for gen in range(n_gen):
        parents = algo.select(pop, algo.pop_size)      # who mates
        offspring = algo.vary(parents)                 # produce children
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = algo.survive(pop, offspring)             # who carries forward
        algo.on_generation(gen, pop)
    return pop
```

The driver supplies an evaluated population each generation; `select`, `vary`, `survive`, and
`on_generation` are the empty hooks the algorithm fills.

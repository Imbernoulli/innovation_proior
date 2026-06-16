# Context: approximating the Pareto front of a multi-objective problem

## Research question

A multi-objective optimization problem (MOP) asks to minimize `m` conflicting objectives at
once, `F(x) = (f_1(x), ..., f_m(x))` over a feasible set `Omega`. Because the objectives
conflict, no single `x` minimizes all of them; the meaningful answer is the *Pareto set* —
the points `x*` such that no other feasible point is at least as good on every objective and
strictly better on one (these are the non-dominated, or Pareto-optimal, points) — and its
image in objective space, the *Pareto front* (PF). A real decision maker does not want the
entire (often infinite) PF; they want a *manageable, finite, evenly spread* set of
representative trade-off points, computed in a single run, that is both **close to** the true
PF (convergence) and **well distributed along it** (diversity/spread). The PF can be convex,
concave, disconnected, or — with three or more objectives — a curved surface, so the method
must handle all of these geometries.

The precise goal is a single algorithm that simultaneously: (1) drives the whole population
toward the PF with strong, sustained selection pressure even after the population becomes
mutually non-dominated; (2) keeps the approximation evenly spread without an expensive,
delicately-tuned, bolt-on diversity device; (3) costs little per generation (the dominant
machinery of the day is super-linear in the population size); (4) works for two objectives
and for three or more, including nonconvex and disconnected fronts; and (5) is able to make
use of the large body of single-objective and mathematical-programming theory rather than
discarding it. Each existing method below achieves a subset of these; none achieves all at
once. Closing that gap is the problem.

## Background

By the mid-2000s evolutionary algorithms (EAs) are the dominant tool for approximating the
PF, because a population naturally carries many candidate trade-offs at once and the
objectives need not be differentiable or convex. The field state and the load-bearing
concepts:

- **Pareto dominance and the partial order.** `x` dominates `y` iff `f_i(x) <= f_i(y)` for
  all `i` and strictly for at least one. Dominance is the bedrock comparison in multi-objective
  EAs, but it is only a *partial* order: two points can be mutually non-dominated and hence
  incomparable. A diagnostic observed about dominance-based methods is exactly this — once the
  population becomes mostly mutually non-dominated (which happens early, and almost immediately
  when `m` is large, since the fraction of incomparable pairs grows fast with the number of
  objectives), dominance stops separating individuals and the pressure that pushes the front
  toward optimality weakens. Diversity then has to be supplied by a *second*, separate
  mechanism.

- **Scalarization — the classical decomposition theory (e.g. Miettinen, *Nonlinear
  Multiobjective Optimization*).** A cornerstone of traditional (non-evolutionary)
  multi-objective optimization: under mild conditions, a Pareto-optimal solution is the optimum
  of *some* scalar problem whose single objective aggregates all the `f_i`. Approximating the
  PF can therefore be cast as solving a *collection* of scalar problems, each tied to a weight
  vector `lambda = (lambda_1, ..., lambda_m)`, `lambda_i >= 0`, `sum_i lambda_i = 1`. The
  standard aggregation functions, with their classical properties:
  - **Weighted sum:** `g^ws(x | lambda) = sum_i lambda_i f_i(x)`. Simple and smooth, but a
    classical result is that minimizing a weighted sum can reach only supported Pareto points,
    the points exposed by supporting hyperplanes of the attainable objective set. Points lying
    in a nonconvex (re-entrant) region of the PF are not the minimizer of any weighted sum, so
    they are unreachable no matter how the weights are swept.
  - **Tchebycheff (Chebyshev):** `g^te(x | lambda, z*) = max_i { lambda_i |f_i(x) - z*_i| }`,
    where `z* = (z*_1, ..., z*_m)` is the *ideal point*, `z*_i = min_{x in Omega} f_i(x)`. A
    classical theorem: for every Pareto-optimal `x*` there exists a weight `lambda` such that
    `x*` is an optimum of `g^te(. | lambda, z*)`; with positive weights, every minimizer is at
    least weakly Pareto-optimal, and a unique minimizer is Pareto-optimal. So sweeping `lambda`
    can in principle trace the *entire* PF, convex or not.
    Its only blemish is that the `max` makes `g^te` non-smooth, which matters for
    gradient-based solvers but not for derivative-free EAs.
  - **Boundary intersection (Das & Dennis 1998, Normal Boundary Intersection; Messac et al.
    Normalized Normal Constraint).** Geometrically, shoot a set of lines from the reference
    point in the weight directions and find where each line meets the boundary of the
    attainable objective set; evenly spread lines give evenly spread PF points, including on
    nonconvex fronts. The clean version carries an equality constraint `F(x) - z* = d*lambda`;
    a penalty-based variant replaces it with a penalty term — `g^pbi(x | lambda, z*) = d_1 +
    theta*d_2`. With the normalized direction `u = lambda / ||lambda||`,
    `d_1 = |(F(x) - z*)^T u|` is the distance along the line and
    `d_2 = ||(F(x) - z*) - d_1*u|| = ||F(x) - (z* + d_1*u)||` is the perpendicular distance
    off it, with `theta > 0`.
    For three or more objectives, the line-intersection geometry is expected to place points
    more uniformly than Tchebycheff contours with the same weights, at the cost of tuning
    `theta` (too large or too small both degrade it).

- **Evenly spread weight vectors (Das & Dennis simplex-lattice construction).** A standard
  way to place `lambda` uniformly on the unit simplex: take every vector whose components are
  drawn from `{0/H, 1/H, ..., H/H}` and sum to 1. This yields `N = C(H + m - 1, m - 1)`
  vectors, evenly distributed. The count is fixed by `m` and the granularity `H`.

- **Real-coded variation operators (the NSGA-II C-code lineage).** For continuous decision
  variables, the standard recombination/mutation pair:
  - **Simulated binary crossover (SBX; Deb & Agrawal 1995).** Produces two children spread
    around the two parents by sampling a spread factor `beta_q` from a polynomial distribution
    governed by a distribution index `eta_c`; large `eta_c` keeps children near the parents,
    small `eta_c` spreads them out. The bounded form respects the variable box `[x_l, x_u]`.
  - **Polynomial mutation (Deb & Goyal).** Perturbs each variable by `delta_q` drawn from a
    polynomial distribution with distribution index `eta_m`, again box-respecting. Applied per
    variable with probability `p_m`.
  Common defaults across the field: `eta_c = eta_m = 20`, crossover probability `p_c = 1.0`,
  mutation probability `p_m = 1/n` (`n` = number of decision variables), so about one variable
  mutates per individual on average.

- **The ideal/reference point.** `z*_i = min_{x in Omega} f_i(x)` is generally unknown during
  the search (computing it exactly would mean solving `m` single-objective problems), so the
  running minimum of each `f_i` seen so far is the natural cheap substitute.

## Baselines

These are the prior methods a new multi-objective EA would be measured against and would react
to.

**NSGA-II (Deb, Pratap, Agarwal & Meyarivan, *IEEE TEC* 6(2), 2002).** The most successful
multi-objective EA of its time. At generation `t`, from a parent population `P_t` of size `N`
it creates an offspring population `Q_t` (binary-tournament selection + SBX + polynomial
mutation), combines `R_t = P_t ∪ Q_t` (size `2N`), and applies *fast non-dominated sorting*
to partition `R_t` into fronts `F_1, F_2, ...` (`F_1` = non-dominated, `F_2` = non-dominated
after removing `F_1`, etc.), in `O(m N^2)` time. The next population is filled front by front;
the last front that does not fit entirely is truncated by **crowding distance** — for each
objective, sort the front, give the two extreme points infinite distance, and give each
interior point the sum over objectives of the normalized gap between its two neighbors,
`sum_i (f_i^{next} - f_i^{prev}) / (f_i^{max} - f_i^{min})`; keep the most spread-out points.
**Gap:** selection rests on a *partial* order — once the front is full of mutually
non-dominated points, dominance no longer separates them and the push toward the PF relies on
the crowding tie-break, a *separate* density heuristic stapled on for diversity rather than an
intrinsic property; the fast sort is `O(m N^2)` per generation; and because an individual is
never tied to any scalar subproblem, none of the single-objective or scalarization theory can
be brought to bear.

**SPEA2 (Zitzler, Laumanns & Thiele, EUROGEN 2001 / TIK-Report 103) and PAES (Knowles &
Corne).** Other strong non-decomposition methods: SPEA2 assigns each individual a strength-based
fitness plus a `k`-nearest-neighbor density estimate (`k = sqrt(N + |archive|)`) and maintains
an external archive; PAES is a (1+1) evolution strategy with an adaptive grid for density.
**Gap:** same family limitation — they treat the MOP as a whole and rank by dominance plus a
bolt-on density device; the partial-order weakening and the inability to reuse scalar machinery
persist; SPEA2's density computation is itself costly.

**MOGLS — multi-objective genetic local search (Jaszkiewicz, *EJOR* 137, 2002; building on
Ishibuchi & Murata).** Closer in spirit to scalarization: at each iteration it draws a weight
vector `lambda` *at random*, forms a temporary elite population (TEP) of the `K` best current
solutions under the Tchebycheff (or weighted-sum) aggregation `g(. | lambda)`, recombines and
locally-improves two of them, and inserts the result into a current set `CS` that is allowed
to grow large (in reported settings `|CS|` reaches 3000-7000), with an external archive `EP`
of non-dominated points. **Gap:** because a *fresh random* weight is drawn every iteration,
the search effort is spread thinly over effectively infinitely many subproblems rather than
concentrated on a fixed, finite set of representatives the decision maker actually wants; and
forming the TEP costs `O(K |CS|)` per iteration with `|CS|` in the thousands, so each step is
expensive. There is no fixed roster of subproblems that each retain their own current best.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Continuous bi-objective ZDT suite (Zitzler, Deb & Thiele 2000):** ZDT1 (convex front),
  ZDT2 (nonconvex), ZDT3 (disconnected front), ZDT4 (many local fronts), ZDT6 (nonuniform
  density), typically `n = 10`-`30` decision variables in `[0,1]^n` (ZDT4 with one variable in
  `[-5,5]`).
- **Continuous 3-objective DTLZ suite (Deb, Thiele, Laumanns & Zitzler 2002):** DTLZ1 (linear
  front with many local fronts), DTLZ2 (spherical/concave front), with `n` chosen per instance.
- **Multi-objective 0-1 knapsack problem (MOKP):** `m` knapsacks, `n` items, maximize total
  profit per knapsack subject to capacity; the nine standard instances of Zitzler & Thiele
  with `m in {2,3,4}`, `n in {250,500,750}`; needs a greedy repair heuristic for infeasible
  bit-strings.
- **Metrics:** *Set Coverage* `C(A,B)` = fraction of `B` dominated by
  some point in `A`; *distance from representatives* `D(A, P*)` = average distance from a dense
  reference set `P*` on the true PF to its nearest point in the approximation `A` (measures
  convergence + spread together); hypervolume; spread/uniformity. Multiple seeds; identical
  population size and number of function evaluations across compared algorithms; SBX/PM with
  `eta = 20`, `p_c = 1.0`, `p_m = 1/n`.

## Code framework

A new method plugs into the same generation-based multi-objective EA harness already used for
the baselines. The MOP, the box-bounded real decision space, and the standard real-coded
operators (SBX crossover, polynomial mutation) already exist as library primitives, as do
baseline selection and sorting utilities. What is *not* settled is how the algorithm
chooses who mates, how it produces offspring, and how it decides which individuals
survive from one generation to the next; those are the slots to be designed. The
substrate is therefore just the generic per-generation loop plus three empty hooks.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # existing utilities: bounded SBX, polynomial mutation, baseline sorting


class CustomMOEA:
    """Generic multi-objective EA. Owns problem parameters and the standard
    real-coded operators; the selection / variation / survival rules are the
    empty slots to be designed. Each individual exposes ind.fitness.values
    (a tuple of objective values, all minimized) and ind.fitness.dominates(other)."""

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
        pass


# existing generation-based driver the algorithm plugs into
def run(algo, toolbox, n_gen):
    pop = toolbox.init_population(algo.pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)     # evaluate the m objectives
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

The driver supplies an evaluated population each generation; `select`, `vary`, and `survive`
are the three empty hooks the algorithm fills.

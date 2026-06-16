# MOEA/D, distilled

MOEA/D (Multiobjective Evolutionary Algorithm based on Decomposition) approximates the Pareto
front by **decomposing** the multi-objective problem into `N` scalar subproblems — one per
evenly spread weight vector — and optimizing them **simultaneously and cooperatively** in a
single run. Each subproblem keeps exactly one current solution (the population is the roster of
these `N` bests, one per weight), and subproblems whose weight vectors are close exchange
information through a **neighborhood** structure. Because the weights are evenly spread and
each subproblem has a total scalar order, MOEA/D gets sustained scalar selection pressure and
spread pressure *intrinsically*, with no Pareto sorting and no separate diversity device, at lower
per-generation cost than dominance-based methods.

## Problem it solves

Find a finite, evenly spread, well-converged approximation to the Pareto front of a
multi-objective problem `min F(x) = (f_1(x), ..., f_m(x))` over `x in Omega`, in one run,
handling convex, nonconvex, and disconnected fronts and `m >= 3` objectives, while keeping
strong selection pressure (which Pareto dominance loses once the population is mutually
non-dominated) and avoiding an expensive bolt-on diversity mechanism.

## Key idea

Pick `N` evenly spread weight vectors `lambda^1, ..., lambda^N` once. Attach a scalar
subproblem to each, with a scalar aggregation function `g(x | lambda, z*)`. Keep one current
solution per subproblem. Define each subproblem's neighborhood `B(i)` = the `T` subproblems
with the nearest weight vectors. Mate within the neighborhood (close weights => close optima
=> good mates) and let each offspring update every neighbor it improves.

- **Total order per subproblem** => selection never goes silent the way Pareto dominance does
  once everything is mutually non-dominated.
- **Even weights => spread pressure is built in.** Each subproblem targets a different
  trade-off direction, so the population is organized across the front by construction —
  no crowding distance, no density grid, no `k`-NN.
- **Neighborhood cooperation** replaces the global non-dominated sort. Per generation:
  `O(mNT)` vs NSGA-II's `O(m N^2)`; ratio `O(T)/O(N)`, smaller since `T << N`. Extra memory
  over NSGA-II is only the `O(m)` reference point.

## Decomposition (scalarization) functions

Weight `lambda = (lambda_1, ..., lambda_m)`, `lambda_i >= 0`, `sum_i lambda_i = 1`; ideal
point `z*_i = min_{x in Omega} f_i(x)` (in practice the running per-objective minimum `z`).

- **Weighted sum:** `g^ws(x | lambda) = sum_i lambda_i f_i(x)`. Smooth but reaches only
  supported Pareto points on the boundary of the convex hull of the attainable objective set;
  points in nonconvex regions are unreachable. Good when the front is convex.
- **Tchebycheff (default):** `g^te(x | lambda, z*) = max_i { lambda_i |f_i(x) - z*_i| }`. For
  every Pareto-optimal `x*` there is a `lambda` making `x*` a minimizer; when all gaps
  `f_i(x*) - z*_i` are positive, take `lambda_i ∝ 1/(f_i(x*) - z*_i)`, and zero-gap cases are
  handled as the corresponding limiting/weak-weight case. With positive weights, every
  minimizer is weakly Pareto-optimal, and a unique minimizer is Pareto-optimal — so it reaches
  the *whole* front, convex or not. Non-smooth, but a derivative-free EA never differentiates it.
- **PBI (for `m >= 3` when uniformity matters):** with `u = lambda / ||lambda||`,
  `g^pbi(x | lambda, z*) = d_1 + theta d_2`, where
  `d_1 = |(F(x) - z*)^T u|` is distance along the line through `z*` in direction `lambda` and
  `d_2 = ||(F(x) - z*) - d_1 u|| = ||F(x) - (z* + d_1 u)||` is perpendicular distance off it,
  `theta > 0`. The line-projection geometry spreads many-objective fronts more uniformly than
  Tchebycheff, at the cost of tuning `theta` (`5` is a common penalty value).

## Weight vectors and neighborhood

- **Weights (Das-Dennis simplex-lattice):** all vectors with entries in
  `{0/H, 1/H, ..., H/H}` summing to 1; count `N = C(H + m - 1, m - 1)`. For `m = 2` this is the
  even fan `lambda = (i/(N-1), 1 - i/(N-1))`.
- **Neighborhood `B(i)`:** the `T` weight-nearest subproblems by Euclidean distance in weight
  space; `i in B(i)` (each weight is its own nearest neighbor). `T = 20` is standard. Too small
  => parents nearly identical, exploration stalls; too large => mates from distant subproblems,
  exploitation weakens and the `O(mT)` update cost climbs.

## Final algorithm (Tchebycheff version)

```
Input: MOP F; weights lambda^1..lambda^N; neighborhood size T; stopping criterion.
1. Initialization
   1.1  EP <- {}                                   # external archive (optional)
   1.2  for each i: B(i) <- indices of the T nearest weights to lambda^i   # includes i
   1.3  x^1..x^N <- random initial solutions;  FV^i <- F(x^i)
   1.4  z <- per-objective minimum of FV           # running ideal point
2. Update — for i = 1..N:
   2.1  pick k, l at random from B(i)              # (with prob 1-delta, from all of 1..N)
   2.2  y <- variation(x^k, x^l)                   # SBX + polynomial mutation, one child
   2.3  for j = 1..m: if f_j(y) < z_j: z_j <- f_j(y)        # update ideal
   2.4  for each j in B(i):                                  # neighbor update
            if g^te(y | lambda^j, z) < g^te(x^j | lambda^j, z):
                x^j <- y;  FV^j <- F(y)
   2.5  update EP with F(y)                         # (optional)
3. Stop when the criterion holds; output EP (or the final population x^1..x^N).
```

`delta` = probability of drawing parents from `B(i)` rather than the whole population.
`delta = 1.0` gives the strict neighborhood loop; the implementation below uses `0.9`, mostly
local mating with an occasional global draw.
**Comparison variant vs NSGA-II:** drop `EP` (return the final `x^1..x^N`), drop any repair,
keep only `z` as extra `O(m)` state.

- **Operators:** simulated binary crossover (distribution index `eta_c = 20`, probability
  `1.0`) + polynomial mutation (distribution index `eta_m = 20`, per-variable probability
  `p_m = 1/n`), the standard real-coded NSGA-II pair; one offspring per mating.
- **Complexity:** `O(mNT)` per generation (`O(1)` parent pick + `O(m)` ideal update + `O(mT)`
  neighbor update, over `N` passes), vs NSGA-II's `O(m N^2)`.

## Working code

Filling the `select` / `vary` / `survive` slots of the generation-based multi-objective EA
harness, using the standard DEAP real-coded operators and a clean numpy MOEA/D core. The
roster of one-solution-per-subproblem is the population; mating happens within neighborhoods;
the reference-point update and decomposition-based replacement are in `survive`.

```python
import random
from copy import deepcopy
from math import comb

import numpy as np
from scipy.spatial.distance import cdist
from deap import tools   # cxSimulatedBinaryBounded, mutPolynomialBounded, uniform_reference_points


class CustomMOEA:
    """MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.
    Decomposes the MOP into N scalar subproblems on evenly spread weight
    vectors; each subproblem keeps one current solution and cooperates only
    with its T weight-nearest neighbors."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index
        self.mut_eta = mut_eta                     # polynomial-mutation distribution index
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.delta = 0.9                           # prob. of mating within the neighborhood
        self.theta = 5.0                           # PBI penalty

        # N evenly spread weight vectors on the unit simplex (Das-Dennis lattice).
        self.H = self._lattice_resolution(pop_size, n_obj)
        self.weights = self._generate_weights(pop_size, n_obj, self.H)
        self.pop_size = len(self.weights)          # actual N from the lattice count
        self.T = min(20, self.pop_size)            # neighborhood size

        # B(i): the T weight-nearest subproblems (argsort over pairwise weight
        # distances; column 0 is i itself, so each B(i) contains i).
        dist = cdist(self.weights, self.weights)
        self.neighbors = [np.argsort(dist[i])[:self.T].tolist()
                          for i in range(self.pop_size)]

        self.z_star = None                         # running ideal point z (per-objective min)
        self._offspring_sources = []

    def _lattice_resolution(self, n, n_obj):
        if n_obj == 2:
            return max(n - 1, 1)                   # C(H+1,1) = H+1 = N
        H = 1
        while comb(H + n_obj - 1, n_obj - 1) < n:  # N = C(H+m-1,m-1)
            H += 1
        return H

    def _generate_weights(self, n, n_obj, H):
        if n_obj == 2:                             # even fan of bi-objective directions
            if n <= 1:
                return np.array([[0.5, 0.5]])
            return np.array([[i / max(n - 1, 1), 1.0 - i / max(n - 1, 1)]
                             for i in range(n)])
        # m >= 3: simplex-lattice points, N = C(H+m-1, m-1).
        return np.array(tools.uniform_reference_points(n_obj, p=H), dtype=float)

    def _tchebycheff(self, fvals, weight, z):
        # g^te = max_j lambda_j * |f_j - z_j|
        return max(weight[j] * abs(fvals[j] - z[j]) for j in range(self.n_obj))

    def _pbi(self, fvals, weight, z):
        # PBI distance: project onto the normalized weight direction.
        diff = np.asarray(fvals, dtype=float) - np.asarray(z, dtype=float)
        w = np.asarray(weight, dtype=float)
        norm = np.linalg.norm(w)
        u = w / norm
        d1 = float(abs(np.dot(diff, u)))
        d2 = float(np.linalg.norm(diff - d1 * u))
        return d1 + self.theta * d2

    def _decompose(self, fvals, weight, z):
        if self.n_obj <= 2:
            return self._tchebycheff(fvals, weight, z)
        return self._pbi(fvals, weight, z)

    def select(self, population, k):
        # No global parent selection: the roster IS the population, one solution
        # per subproblem. Hand it back for neighborhood mating in vary().
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        # One offspring per subproblem i, from two parents drawn (with prob delta)
        # from B(i) -- close subproblems have close optima, so neighbors mate well;
        # with prob 1-delta mate globally to escape local fronts.
        offspring = []
        lo, hi = self.bounds
        order = random.sample(range(len(parents)), len(parents))
        self._offspring_sources = order
        for i in order:
            if random.random() < self.delta:
                pool = [parents[j] for j in self.neighbors[i]]
            else:
                pool = parents
            if len(pool) < 2:
                pool = parents
            a, b = random.sample(range(len(pool)), 2)
            child = deepcopy(pool[a])
            mate = deepcopy(pool[b])
            tools.cxSimulatedBinaryBounded(child, mate, eta=self.cx_eta, low=lo, up=hi)  # p_c = 1
            tools.mutPolynomialBounded(child, eta=self.mut_eta, low=lo, up=hi,
                                       indpb=self.mut_prob)
            del child.fitness.values               # invalidate so the harness re-evaluates
            offspring.append(child)
        return offspring

    def survive(self, population, offspring):
        # The heart: initialize z from evaluated parents, then for each evaluated
        # offspring update z and replace the neighboring subproblems it beats.
        next_pop = list(population)
        if not next_pop:
            return population
        if self.z_star is None:
            self.z_star = [float('inf')] * self.n_obj
            for ind in next_pop:
                if ind.fitness.valid:
                    for j in range(self.n_obj):
                        if ind.fitness.values[j] < self.z_star[j]:
                            self.z_star[j] = ind.fitness.values[j]

        sources = self._offspring_sources or list(range(len(offspring)))
        for source_i, child in zip(sources, offspring):
            if not child.fitness.valid:
                continue
            for j in range(self.n_obj):             # z_j <- min over all seen f_j
                if child.fitness.values[j] < self.z_star[j]:
                    self.z_star[j] = child.fitness.values[j]
            for j in self.neighbors[source_i]:      # try child on every neighbor j of source_i
                if j >= len(next_pop):
                    continue
                g_child = self._decompose(child.fitness.values, self.weights[j], self.z_star)
                g_cur = self._decompose(next_pop[j].fitness.values, self.weights[j], self.z_star)
                if g_child < g_cur:                 # offspring better for subproblem j -> take over
                    next_pop[j] = deepcopy(child)
        return next_pop[:self.pop_size]

    def on_generation(self, gen, population):
        pass
```

Clean numpy core: neighbors via
`argsort(cdist(weights, weights))[:, :T]`, ideal point `z = min` over evaluated `F`, Tchebycheff
for `m <= 2`, PBI for `m >= 3`, and neighbor replacement whenever
`g(offspring) < g(incumbent)`. To use an archive, add an external population `EP` updated with
each offspring's `F` and return `EP`; the internal update loop stays the same.

# Context: Decomposing a multi-objective problem into co-evolving scalar subproblems

## Research question

A great many design problems carry several conflicting objectives at once — minimize cost *and* maximize reliability, minimize weight *and* maximize stiffness. There is no single best solution: improving one objective costs another. The honest answer is a *set* — the Pareto-optimal trade-off surface, the points that cannot be improved in one objective without being worsened in another. A decision-maker wants to see this whole surface and then pick.

A population-based evolutionary algorithm is the natural vehicle for producing the whole surface in one run, because its unit of work is already a *set* of candidates. The dominant way to drive such a population is to rank its members by Pareto dominance — give better-front solutions higher fitness, and add a density device to spread them along the front. That works, but it leans entirely on a *partial* order. As the number of objectives grows, almost every pair of solutions becomes mutually nondominated, so dominance stops separating the population: nearly everyone lands on the first front and the selection that is supposed to drive convergence degrades toward random. The sorting step is also quadratic per generation, and the diversity is governed by a geometric heuristic that has no direct link to a stated "spread evenly across the front" target.

The question this landscape poses: can a single run return a good approximation of the *entire* Pareto front — well-converged and evenly spread — while keeping a *decisive* (total-order) selection pressure that does not weaken as objectives are added, and a per-generation cost cheaper than a global nondominated sort?

## Background

**Pareto dominance.** For a minimization problem with objective vector f(x) = (f₁, …, f_M), solution x *dominates* y iff fᵢ(x) ≤ fᵢ(y) for every i and f_j(x) < f_j(y) for at least one j. A solution is *nondominated* in a set if nothing dominates it; its image is a point on the Pareto front. A good approximation needs both **convergence** (points on/near the true front) and **diversity** (an even spread, including the extremes). Dominance is only a *partial* order: most pairs are incomparable, which is exactly what a front is, but also why ranking a population by dominance loses resolution as M grows.

**Classical scalarization.** The mathematical-programming tradition (Miettinen, *Nonlinear Multiobjective Optimization*, 1999) turns a multi-objective problem into a single scalar one by aggregating the objectives with a weight vector w. Under mild conditions a Pareto-optimal solution is the optimum of such a scalar subproblem, and conversely the optimum of a suitable scalar subproblem is Pareto-optimal. Three aggregations are standard:
- **Weighted sum:** g(x|w) = Σᵢ wᵢ fᵢ(x), with wᵢ ≥ 0 and Σ wᵢ = 1. Minimizing it slides a hyperplane with normal w down until it touches the feasible objective set — it touches the *convex* boundary. On a non-convex (concave) front, no choice of nonnegative weights ever lands a solution in a concave dip; the hyperplane skips across it. So a weighted sum is blind to concave regions of the front.
- **Tchebycheff (weighted Chebyshev / L_∞):** g(x|w, z*) = max_i wᵢ |fᵢ(x) − zᵢ*|, where z* is the ideal point (zᵢ* = min over the feasible region of fᵢ). Its contours in objective space are nested "L-shaped" (axis-aligned corner) sets rather than hyperplanes, so they can touch the front in concave regions too. For any Pareto-optimal point there is a weight that makes it the unique minimizer, and varying w sweeps the whole front — convex *and* concave. The price is non-smoothness (a max).
- **Boundary-intersection / Normal-Boundary-Intersection (Das & Dennis, 1998):** instead of weighting objectives, push a solution along a reference direction toward the front and measure how far it travels and how far it strays off the line. This gives an even spread of front points by construction, at the cost of an extra equality constraint (handled later with a penalty).

**Uniform weight vectors on the simplex.** Das & Dennis (1998) also gave the structured way to lay down weights: divide each coordinate into H equal segments and take every vector whose nonnegative components are multiples of 1/H summing to 1. This produces a simplex-lattice of N = C(H+M−1, M−1) evenly spread weight vectors. For two objectives this is simply w = (i/H, 1−i/H), i = 0…H, giving N = H+1 points.

**Why one scalarization is not enough — and why a population changes the accounting.** Classical scalarization recovers a front by *re-solving* the scalar subproblem for many weight settings — one full optimization per point on a curve you want densely sampled. That is the standard objection to scalarization: it is expensive, one run per point. But that accounting assumes the subproblems are solved *independently and sequentially*. A population holds many candidate solutions at once; if each member is tied to its own weight vector, the "many runs" collapse into a single co-evolving run.

**Neighboring subproblems share structure.** Two weight vectors that are close in weight space define two scalar subproblems with nearly the same aggregation, hence nearly the same optimum. So a solution that is good for one subproblem is, almost certainly, good for its neighbors. This is the pre-method fact that makes co-evolution pay: information (good solutions, recombination material) can be shared between subproblems with nearby weights instead of each subproblem searching in isolation.

## Baselines

**Weighted-sum sweep / ε-constraint (classical).** Pick a weight (or an objective bound), solve the resulting single-objective program, get one Pareto point; repeat over a grid. Recovers the front only point-by-point, and the weighted-sum version misses concave regions entirely. No population, no information sharing across weights.

**VEGA — Vector Evaluated GA (Schaffer, 1985).** Split the population into M sub-populations, select sub-population m purely by objective m, shuffle, recombine. Each sub-population is effectively a single-objective (corner-weight) selector, so the population speciates toward the M objective extremes and starves the middle of the front; with no notion of trade-off compromise it tends to miss the interior of the front (Schaffer noted the difficulty with non-convex fronts), and it has no concept of dominance.

**MOGA / NSGA — dominance-ranked GAs (Fonseca & Fleming 1993; Srinivas & Deb 1994).** Bring Pareto dominance into fitness: rank by how many members dominate you, or sort the population into successive nondomination fronts, and spread each front with fitness sharing (which needs a niche radius σ_share). Genuine convergence pressure, but the ranking is a partial order whose resolution falls as M grows, and the sort is costly (the original NSGA's is O(MN³)).

**NSGA-II — fast elitist dominance ranking (Deb et al.).** The strong dominance-based baseline. A fast nondominated sort (compute every pairwise dominance once; peel fronts by decrementing domination counts) brings the per-generation sort to O(MN²); elitism comes from merging parents and offspring (size 2N) and refilling the next generation front by front; diversity is a parameter-free crowding distance (the normalized side-length of the cuboid spanned by a solution's nearest neighbors along the front, infinite at the extremes). Selection fuses the two by a lexicographic crowded-comparison: better front first, larger crowding distance on a tie. It removed σ_share and the cubic sort, but the convergence signal is still the *partial* dominance order — its selection pressure thins as objectives are added — and the O(MN²) sort plus a heuristic crowding measure remain the per-generation engine.

**MOGLS — Multi-Objective Genetic Local Search (Ishibuchi & Murata 1996; Jaszkiewicz 2002).** A scalarization-based EA: at each step draw a (often random) weight vector, build a temporary mating pool of solutions that are good for that weight, recombine, and apply local search under that scalarizing function. It already uses scalarization inside an EA, but the weight is resampled per iteration rather than fixed and structured across a standing population, so there is no persistent one-solution-per-subproblem bookkeeping and no explicit neighborhood among a fixed set of weights.

## Evaluation settings

Two-objective and three-objective test problems from the literature would be the natural yardstick: Schaffer's SCH, the ZDT family (ZDT1 convex, ZDT2 concave, ZDT3 disconnected, ZDT4 multimodal, ZDT6 non-uniformly dense), and the scalable DTLZ family (DTLZ1–DTLZ2) for three or more objectives. ZDT1 in particular is m = 2, n = 30 decision variables on [0,1], with objectives f₁ = x₁, g = 1 + 9·(Σ_{i≥2} xᵢ)/(n−1), f₂ = g·(1 − √(f₁/g)), and a convex true front f₂ = 1 − √f₁ reached when xᵢ = 0 for i ≥ 2. A run is scored on a **convergence** metric (mean distance from the obtained set to the true front) and a **spread/diversity** metric (how evenly the obtained points cover the front, including the extremes), or jointly by a metric such as IGD against a reference sampling of the true front. The natural points of comparison are the dominance-based elitist MOEAs run with their recommended settings. Solutions are real-coded with simulated binary crossover (distribution index η_c) and polynomial mutation (index η_m).

## Code framework

The available primitives are a problem that maps a batch of decision vectors to objective vectors, the Das & Dennis simplex-lattice weight generator, real-coded variation (simulated binary crossover, polynomial mutation), and a dominance test for maintaining an external nondominated archive. The empty slots are the scalar aggregation that turns one weight vector plus a reference point into a single fitness, the neighborhood structure over weight vectors, and the main loop that reproduces from a neighborhood and lets an offspring update its neighbors.

```python
import numpy as np

def dominates(f_a, f_b):
    # minimization: a dominates b iff a <= b in all and a < b in at least one
    return np.all(f_a <= f_b) and np.any(f_a < f_b)

def das_dennis_weights(n_partitions, n_obj):
    # uniform simplex lattice: components are multiples of 1/H summing to 1.
    # N = C(H + n_obj - 1, n_obj - 1) weight vectors. (exists)
    ...

def sbx_crossover(p1, p2, xl, xu, eta, pc, rng):   # simulated binary crossover (exists)
    ...
def polynomial_mutation(x, xl, xu, eta, rng):      # polynomial mutation (exists)
    ...

def evaluate(problem, X):                          # X -> objective matrix F (exists)
    ...

def aggregate(F, w, z):
    # TODO: collapse an objective vector (or batch) into ONE scalar fitness for the
    #       subproblem with weight w and reference point z. Must give a TOTAL order,
    #       and must be able to reach concave regions of the front.
    pass

def neighborhoods(W, T):
    # TODO: for each weight vector, the indices of its T nearest weight vectors
    #       (Euclidean in weight space). Cheap O(N^2) once, reused every generation.
    pass

def update_reference_point(z, f_y):
    # TODO: keep z as the running best (ideal) per objective.
    pass

def run(problem, n_partitions, T, n_gen):
    # TODO: one solution per weight vector; reproduce from a neighborhood;
    #       let the offspring update the subproblems in that neighborhood;
    #       maintain an external nondominated archive to report.
    pass
```

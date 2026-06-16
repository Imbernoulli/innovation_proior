# NSGA-III, distilled

NSGA-III is a reference-point-based many-objective evolutionary algorithm. It keeps NSGA-II's
elitist framework — combine the `N` parents with `N` offspring, fast-non-dominated-sort the
`2N` pool, fill the next population front by front — but **replaces the crowding-distance
split of the last front with reference-point niching**. Diversity is supplied externally by a
fixed grid of well-spread reference directions and enforced by filling the least-represented
direction first, instead of being estimated from the population's own density. This is what
targets many-objective cases where crowding distance becomes both uninformative (almost
everything is mutually non-dominated; per-axis neighbor gaps stop measuring real crowding)
and expensive.

## Problem it solves

Approximate the whole Pareto front of a many-objective problem `min f(x) = (f_1, ..., f_M)`,
`M >= 4`, in one run — converged onto the front and uniformly spread across it — while keeping
NSGA-II's elitism and fast non-dominated sorting and introducing no fragile diversity
parameter.

## Key ideas

- **External, fixed diversity targets (Das-Dennis reference points).** Lay `H = C(M+p-1, p)`
  evenly spaced points on the unit simplex `sum_i w_i = 1` (components in `{0/p, ..., p/p}`).
  The ray from the origin through each is a *reference line* (direction). For large `M`,
  combine a boundary layer and an inner layer (smaller `p`, scaled inward) so the interior
  isn't starved — the fraction of single-layer Das-Dennis points in the simplex interior
  is `C(p-1, M-1) / C(M+p-1, M-1)` when `p >= M` and zero when `p < M`.
- **Adaptive normalization every generation.** Re-fit the grid to the current survivors `S_t`:
  ideal point `z_i^min = min_{S_t} f_i`; per-axis extreme via the achievement scalarizing
  function `ASF(s, w^j) = max_i (f_i(s) - z_i^min)/w_i^j` with `w_j^j = 1` and off-axis
  weights `1e-6`, so the axis-`j` extreme is a true corner (large in `j`, small elsewhere),
  not a dominated outlier; axis-intercept scales from the hyperplane through the translated
  extremes (`A x = 1`, `A = extremes - ideal`, `a_j = 1/x_j`, with fallbacks on degeneracy);
  normalized objective `f_i^n = (f_i - z_i^min)/a_i` in translated coordinates, equivalently
  `(f_i - z_i^min)/(absolute_intercept_i - z_i^min)` when storing absolute intercepts.
- **Association by perpendicular distance to a line, not a point.** A reference point is a
  *direction*; a solution serves it well if it lies near the ray regardless of how far out.
  Associate `s` to the line minimizing `d_perp(s, w) = || s^n - ((s^n·w)/||w||^2) w ||`. This
  factors convergence (position along the ray, handled by Pareto rank) from diversity (which
  ray, handled by the split).
- **Niche-preservation split.** Let `rho_j` = number of already-selected members (fronts
  `1..l-1`) associated to reference point `j`. To fill `K` slots from the last front `F_l`,
  repeatedly pick the reference point with the smallest `rho_j` (ties random): if `rho_j = 0`,
  add the associated `F_l` member with the shortest `d_perp` (best ambassador for a new niche);
  if `rho_j >= 1`, add a random associated member; if none is associated, drop that reference
  point for this generation. Increment `rho_j`, repeat `K` times.

## Parameters and why

- **Reference points `H` and population `N ~ H`** are the user's resolution choice, not new
  algorithmic knobs — NSGA-III is parameter-less in the same sense as NSGA-II (no
  `sigma_share`, no MOEA/D `theta`/`T`). `N` is kept a multiple of four.
- **SBX**: crossover probability `p_c = 1`, distribution index `eta_c = 30` — deliberately
  *large*, to keep children near their parents (a soft mating restriction). In a sparse
  many-objective population, distant parents would otherwise produce offspring distant from
  both, landing in unexplored voids; a large `eta_c` recombines locally and keeps offspring
  useful.
- **Polynomial mutation**: `eta_m = 20`, `p_m = 1/n` (about one of `n` variables per
  individual).
- **No mating selection.** Parents are drawn at random from `P_{t+1}`. Because survival
  niching already spreads the population uniformly across reference directions, mating need
  not also enforce spread; the diversity work lives entirely in survival.

## Algorithm (one generation)

```
Input: H reference points Z, parent population P_t (size N)
1. Q_t = SBX + polynomial mutation on random parents from P_t
2. R_t = P_t ∪ Q_t                                     # elitist, size 2N
3. F_1, F_2, ... = fast_non_dominated_sort(R_t)
4. S_t = {}; add fronts F_1, F_2, ... until |S_t| >= N; let F_l be the last added
5. if |S_t| == N: return S_t
6. P_{t+1} = F_1 ∪ ... ∪ F_{l-1};   K = N - |P_{t+1}|
7. normalize S_t:  ideal z_min;  extremes via ASF;  intercept scales a;  f^n = (f - z_min)/a
8. associate each s in S_t to argmin reference line by perpendicular distance d_perp
9. rho_j = #(members of P_{t+1} associated to ref j)
10. niching: K times, take the smallest-rho_j reference point and add a member of F_l
    (closest if rho_j==0, else random); update rho_j
11. return P_{t+1}
```

## Cost

Non-dominated sort `O(N log^{M-2} N)`; normalization `O(M^2 N)` extremes + `O(M^3)` one solve;
association `O(M N H)`; niching `O(N H)`. With `N ~ H` this is `O(N^2 log^{M-2} N)` or
`O(N^2 M)` — the same ballpark as NSGA-II, with no high-dimensional density estimate.

## Working code

The reference-point selection operator (the contribution) and the thin deployment shell. The
selection operator keeps whole fronts and splits the last by niching, replacing crowding
distance.

```python
import numpy
from collections import namedtuple
from itertools import chain


NSGA3Memory = namedtuple("NSGA3Memory", ["best_point", "worst_point", "extreme_points"])


def uniform_reference_points(nobj, p=4, scaling=None):
    """Das-Dennis points on sum w_i = 1; optional scaling shrinks a layer
    toward the simplex center so several layers can be combined."""
    def recurse(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(recurse(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = numpy.array(recurse(numpy.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj
    return ref_points


def find_extreme_points(fitnesses, best_point, extreme_points=None):
    """Per axis j, minimize ASF(s,w^j)=max_i (s_i-best_i)/w_i^j.
    DEAP implements w_i^j=1e-6 off axis as a 1e6 multiplier."""
    if extreme_points is not None:
        fitnesses = numpy.concatenate((fitnesses, extreme_points), axis=0)
    ft = fitnesses - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)
    return fitnesses[numpy.argmin(asf, axis=1), :]


def find_intercepts(extremes, ideal, current_worst, front_worst):
    """Axis-intercept scales of the hyperplane through the M extreme points:
    solve A x = 1 with A = extremes - ideal, then a_j = 1/x_j."""
    b = numpy.ones(extremes.shape[1])
    A = extremes - ideal
    try:
        x = numpy.linalg.solve(A, b)
    except numpy.linalg.LinAlgError:
        return current_worst
    if numpy.count_nonzero(x) != len(x):
        return front_worst
    intercepts = 1 / x
    if (not numpy.allclose(numpy.dot(A, x), b)
            or numpy.any(intercepts <= 1e-6)
            or numpy.any((intercepts + ideal) > current_worst)):
        return front_worst
    return intercepts


def associate_to_niche(F, ref_points, ideal, intercepts):
    """Normalize with DEAP's intercept convention, then assign each member
    to the reference line of minimum perpendicular distance."""
    fn = (F - ideal) / (intercepts - ideal + numpy.finfo(float).eps)
    fn = numpy.repeat(numpy.expand_dims(fn, axis=1), len(ref_points), axis=1)
    norm = numpy.linalg.norm(ref_points, axis=1)
    d = numpy.sum(fn * ref_points, axis=2) / norm.reshape(1, -1)
    proj = d[:, :, numpy.newaxis] * ref_points[numpy.newaxis, :, :] / norm[numpy.newaxis, :, numpy.newaxis]
    dist = numpy.linalg.norm(proj - fn, axis=2)        # perpendicular distance to each line
    niches = numpy.argmin(dist, axis=1)
    dist = dist[range(niches.shape[0]), niches]
    return niches, dist


def niching(last_front, k, niches, distances, niche_counts):
    """Fill k slots from the last front, serving the smallest-niche-count
    reference point first (closest member if empty, else random)."""
    selected = []
    available = numpy.ones(len(last_front), dtype=bool)
    while len(selected) < k:
        live = numpy.zeros(len(niche_counts), dtype=bool)
        live[numpy.unique(niches[available])] = True
        min_count = numpy.min(niche_counts[live])
        chosen_refs = numpy.flatnonzero(live & (niche_counts == min_count))
        numpy.random.shuffle(chosen_refs)
        for j in chosen_refs[: k - len(selected)]:
            members = numpy.flatnonzero((niches == j) & available)
            if niche_counts[j] == 0:
                sel = members[numpy.argmin(distances[members])]   # new niche -> closest
            else:
                numpy.random.shuffle(members)
                sel = members[0]                                  # filled niche -> random
            available[sel] = False
            niche_counts[j] += 1
            selected.append(last_front[sel])
    return selected


def selNSGA3(individuals, k, ref_points, nd="log", best_point=None,
             worst_point=None, extreme_points=None, return_memory=False):
    """DEAP-style environmental selection: keep whole fronts, split the last
    by reference-point niching."""
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception("selNSGA3: invalid non-dominated sorting choice")

    fitnesses = numpy.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1.0                       # always handle selection as minimization

    if best_point is not None and worst_point is not None:
        best_point = numpy.min(numpy.concatenate((fitnesses, best_point), axis=0), axis=0)
        worst_point = numpy.max(numpy.concatenate((fitnesses, worst_point), axis=0), axis=0)
    else:
        best_point = numpy.min(fitnesses, axis=0)
        worst_point = numpy.max(fitnesses, axis=0)

    extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
    front_worst = numpy.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    idx, cnt = numpy.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[idx] = cnt

    chosen = list(chain(*pareto_fronts[:-1]))
    n = k - len(chosen)
    chosen.extend(niching(pareto_fronts[-1], n,
                          niches[len(chosen):], dist[len(chosen):], niche_counts))

    if return_memory:
        return chosen, NSGA3Memory(best_point, worst_point, extreme_points)
    return chosen
```

Deployment shell (NSGA-II framework, reference-point survival):

```python
import random
from copy import deepcopy


class NSGA3:
    """NSGA-III: NSGA-II framework with reference-point niching survival."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=30.0, mut_eta=20.0, mut_prob=None,
                 ref_points=None, ref_divisions=4, ref_scaling=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.bounds = bounds
        self.cx_eta = cx_eta                       # large SBX index -> children near parents
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.ref_points = (ref_points if ref_points is not None
                           else uniform_reference_points(n_obj, ref_divisions, ref_scaling))

    def select(self, population, k):
        sel = [deepcopy(ind) for ind in population]   # no mating selection (diversity in survive)
        random.shuffle(sel)
        return sel[:k]

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            cx_simulated_binary_bounded(offspring[i], offspring[i + 1],
                                        eta=self.cx_eta, low=lo, up=hi)
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for ind in offspring:
            mut_polynomial_bounded(ind, eta=self.mut_eta, low=lo, up=hi,
                                   indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring             # elitist (mu + lambda)
        return selNSGA3(combined, self.pop_size, self.ref_points)
```

## Relation to prior methods

- **NSGA-II** — identical elitist combined-pool / fast-sort / fill-fronts framework; NSGA-III
  swaps only the last-front split: reference-point niching instead of crowding distance.
- **MOEA/D** — shares the idea of predefined spread directions, but uses them only for
  *niching inside a dominance-ranked EA*, with no scalarizing function and no neighborhood or
  penalty parameters; objectives are never aggregated into a scalar subproblem.
- **Das-Dennis** — the simplex point construction is reused as the reference grid; the
  weighted-sum critique is exactly why the grid is used for niching, not for scalarization.

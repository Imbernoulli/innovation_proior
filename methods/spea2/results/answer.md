# SPEA2, distilled

SPEA2 (Strength Pareto Evolutionary Algorithm 2) is an elitist multi-objective evolutionary
algorithm. It assigns every individual a single scalar fitness that combines a *fine-grained
dominance-strength* measure (convergence) with a *k-th nearest neighbor density* tie-breaker
(diversity), and maintains a **fixed-size external archive** that is reduced, when it overflows,
by a truncation operator that removes the most crowded individuals while never discarding the
boundary solutions.

## Problem it solves

Approximate the Pareto-optimal set of a vector objective `f(x) = (f_1, …, f_m)` (all minimized)
in a single run, returning a set that is both close to the true front and uniformly spread over
it, extremes included — for two-objective and many-objective problems alike. The obstacle is
that Pareto dominance is a partial order: with several objectives most individuals are mutually
indifferent, so a fitness based on dominance alone leaves the population tied and the search
directionless, and diversity-based archive reduction tends to throw away the extreme trade-offs.

## Key idea

A two-layer scalar fitness, dominance first and density only as a tie-breaker, plus a
fixed-size boundary-preserving archive.

- **Strength.** Each individual `i` in the combined pool `P_t + P̄_t` gets
  `S(i) = |{ j : i ≻ j }|`, the number of individuals it dominates.
- **Raw fitness.** `R(i) = Σ_{ j ≻ i } S(j)`, the summed strengths of `i`'s dominators.
  `R(i) = 0` exactly for the nondominated front; larger `R` means `i` sits deeper in the
  dominated interior, weighted by how dominant its dominators are. Minimized.
- **Density.** Sort the Euclidean objective-space distances from `i` to all others; let `σ_i^k`
  be the `k`-th smallest, with `k = sqrt(N + N̄)` (Silverman's sqrt-of-sample-size rule). Then
  `D(i) = 1 / (σ_i^k + 2)`. The `+2` keeps the denominator positive and forces `D < 1`;
  even duplicates have `D = 1/2`, so every `R=0` individual has `F<1`, while every dominated
  individual has `R≥1`. Density can refine equal raw-fitness levels without crossing them.
- **Fitness.** `F(i) = R(i) + D(i)`. Dominance is the primary key, density the secondary; and
  `F(i) < 1` is exactly the test for being nondominated.

## Final algorithm (main loop)

```
Inputs: N (population size), N̄ (archive size), T (max generations)
Step 1  Init: random P_0, empty archive P̄_0 = ∅, t = 0
Step 2  Fitness: compute S, R, D, F = R + D over P_t + P̄_t
Step 3  Environmental selection -> P̄_{t+1}, size exactly N̄:
          P̄_{t+1} = { i ∈ P_t + P̄_t : F(i) < 1 }          # all nondominated
          if |P̄_{t+1}| < N̄:  fill with the best-dominated (smallest F)
          if |P̄_{t+1}| > N̄:  truncate (operator below)
Step 4  Termination: if t ≥ T, return nondominated members of P̄_{t+1}; stop
Step 5  Mating: binary tournament with replacement on P̄_{t+1} -> mating pool
Step 6  Variation: recombination + mutation -> P_{t+1}; t = t + 1; go to Step 2
```

**Truncation operator** (when `|P̄_{t+1}| > N̄`): iteratively remove the individual `i` that is
at least as crowded as every other, `i ≤_d j` for all `j`, where

```
i ≤_d j  :⇔  (∀ 0<k<|P̄_{t+1}| : σ_i^k = σ_j^k)
            ∨ (∃ 0<k<|P̄_{t+1}| : (∀ 0<l<k : σ_i^l = σ_j^l) ∧ σ_i^k < σ_j^k)
```

i.e. remove the individual with the smallest distance to its nearest neighbor, ties broken by
the second-nearest, then the third, and so on (lexicographically smallest sorted
neighbor-distance vector). This removes from the densest region; because a boundary solution has
a far nearest neighbor, it is never the minimum, so the extremes are preserved.

Complexity per generation: `S`, `R` are `O(M^2)`; density and truncation are `O(M^2 log M)` on
average (`O(M^3)` worst case for truncation), `M = N + N̄`.

## Why the design choices

- **Strength for *every* individual, not just the archive.** Restricting strength to the archive
  (as in the predecessor SPEA) gives identical fitness to all individuals dominated by the same
  archive set, collapsing to random search when the archive is small. Counting what each
  individual itself dominates restores discrimination.
- **`R` = sum of dominators' strengths, not a plain count.** Being dominated by a high-strength
  individual (one far out toward the front) means sitting deep in the interior; weighting by the
  dominator's strength encodes how far back in the order an individual is.
- **Density as a strict tie-breaker via `+2`.** `D < 1` makes `F<1` exactly the nondominance
  test and lets density reorder equal raw-fitness levels without crossing them. `+1` would let
  a duplicate (`σ = 0`) reach `D = 1`, putting a nondominated duplicate exactly on the strict
  archive threshold; `+2` keeps a margin.
- **k-th NN density** over a hyper-grid histogram (PESA, resolution-dependent) or a per-objective
  crowding distance (NSGA-II, axis-sort-based, single-front): a true metric density, one scalar
  per individual, with bandwidth that adapts to local density.
- **Fixed archive size with fill-on-underflow** keeps selection pressure and the mating-pool size
  steady (the predecessor's archive drifted in size).
- **Truncation by closest-neighbor removal** removes from the densest spot and provably never
  removes the boundary points, unlike clustering, which merges to centroids and drops the
  extremes.

## Working code

Grounded in the canonical DEAP implementation (`deap/tools/emo.py`, `selSPEA2`). The selector
copies nondominated individuals using raw `R<1`, which is equivalent to `F<1` because `D<1`;
computes density only when it must fill an underfull archive; and truncates an overfull archive
by the lexicographic nearest-neighbor rule. DEAP keeps distances squared, which preserves the
nearest-neighbor and truncation orders, and uses `K = sqrt(M)` over the combined pool of size
`M = N + N̄`.

```python
import math
import random
from copy import deepcopy

from deap import tools


def sel_spea2(individuals, k):
    """DEAP-style SPEA2 environmental selection: keep k individuals from a
    combined pool by strength/raw fitness, density fill, and truncation."""
    N = len(individuals)
    M = len(individuals[0].fitness.values)          # number of objectives
    K = math.sqrt(N)                                # k-th neighbour ~ sqrt(sample size)

    strength_fits = [0] * N                          # S(i) = # individuals i dominates
    fits = [0.0] * N                                 # -> R(i), then R(i) + D(i)
    dominating_inds = [[] for _ in range(N)]         # i's dominators

    # Strength S(i) and the dominator lists
    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1:], i + 1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    # Raw fitness R(i) = sum of i's dominators' strengths; R(i)=0 <=> nondominated
    for i in range(N):
        for d in dominating_inds[i]:
            fits[i] += strength_fits[d]

    # Copy all nondominated (R(i) < 1)
    chosen_indices = [i for i in range(N) if fits[i] < 1]

    if len(chosen_indices) < k:                      # underfull -> add density, fill best-dominated
        for i in range(N):
            distances = [0.0] * N
            for j in range(i + 1, N):
                dist = 0.0
                for m in range(M):
                    val = (individuals[i].fitness.values[m]
                           - individuals[j].fitness.values[m])
                    dist += val * val                # squared Euclidean, order-preserving
                distances[j] = dist
            kth_dist = _kth_smallest(distances, 0, N - 1, K)
            fits[i] += 1.0 / (kth_dist + 2.0)        # F(i) = R(i) + D(i), D < 1
        rest = [(fits[i], i) for i in range(N) if i not in chosen_indices]
        rest.sort()
        chosen_indices += [i for _, i in rest[:k - len(chosen_indices)]]

    elif len(chosen_indices) > k:                    # overfull -> truncate densest, keep boundary
        n = len(chosen_indices)
        dist = [[0.0] * n for _ in range(n)]
        order = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = 0.0
                for m in range(M):
                    val = (individuals[chosen_indices[i]].fitness.values[m]
                           - individuals[chosen_indices[j]].fitness.values[m])
                    d += val * val
                dist[i][j] = d
                dist[j][i] = d
            dist[i][i] = -1                           # self is never a neighbour
        for i in range(n):                           # sort each row: sigma_i^1, sigma_i^2, ...
            for j in range(1, n):
                m = j
                while m > 0 and dist[i][j] < dist[i][order[i][m - 1]]:
                    order[i][m] = order[i][m - 1]
                    m -= 1
                order[i][m] = j

        size = n
        to_remove = []
        while size > k:
            min_pos = 0                              # i <=_d j: lexicographically smallest = densest
            for i in range(1, n):
                for j in range(1, size):
                    di = dist[i][order[i][j]]
                    dm = dist[min_pos][order[min_pos][j]]
                    if di < dm:
                        min_pos = i
                        break
                    elif di > dm:
                        break
                    # tie: compare next-nearest neighbour
            for i in range(n):
                dist[i][min_pos] = float("inf")
                dist[min_pos][i] = float("inf")
                for j in range(1, size - 1):
                    if order[i][j] == min_pos:
                        order[i][j], order[i][j + 1] = order[i][j + 1], min_pos
            to_remove.append(min_pos)
            size -= 1
        for idx in reversed(sorted(to_remove)):
            del chosen_indices[idx]

    return [individuals[i] for i in chosen_indices]


def _kth_smallest(array, begin, end, i):
    if begin == end:
        return array[begin]
    q = _partition(array, begin, end)
    cnt = q - begin + 1
    if i < cnt:
        return _kth_smallest(array, begin, q, i)
    return _kth_smallest(array, q + 1, end, i - cnt)


def _partition(array, begin, end):
    p = random.randint(begin, end)
    array[begin], array[p] = array[p], array[begin]
    x = array[begin]
    i, j = begin - 1, end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


class CustomMOEA:
    """SPEA2 strategy: strength-Pareto fitness + k-NN density tie-breaker,
    with a fixed-size, boundary-preserving elite archive."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.archive = []

    def select(self, population, k):
        # Binary tournament on the archive (the elite front); dominance decides a
        # pair, an undecided pair is broken at random.
        pool = self.archive if self.archive else population
        chosen = []
        for _ in range(k):
            a, b = random.sample(pool, 2)
            if a.fitness.dominates(b.fitness):
                chosen.append(deepcopy(a))
            elif b.fitness.dominates(a.fitness):
                chosen.append(deepcopy(b))
            else:
                chosen.append(deepcopy(random.choice((a, b))))
        return chosen

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:               # crossover probability
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
        combined = population + offspring
        survivors = sel_spea2(combined, self.pop_size)
        nd = [ind for ind in survivors
              if not any(o.fitness.dominates(ind.fitness) for o in survivors)]
        self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]
        return survivors

    def on_generation(self, gen, population):
        pass
```

## Relation to prior methods

- **SPEA** (the predecessor): strength assigned only to the archive, fitness too coarse
  (individuals dominated by the same set tie; archive of one ⇒ random search), and archive
  reduced by clustering, which loses boundary solutions and lets the archive size drift. SPEA2
  gives strength to every individual, sums dominator strengths for a fine-grained raw fitness,
  adds k-NN density, fixes the archive size, and replaces clustering with boundary-preserving
  truncation.
- **NSGA-II**: nondomination-rank fronts + per-objective crowding distance. SPEA2 instead folds
  a metric k-NN density into a single scalar with the dominance-strength signal.
- **PESA**: archive-only mating with a hyper-grid (histogram) density. SPEA2 replaces the
  resolution-dependent grid with a k-NN density.

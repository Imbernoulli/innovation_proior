# NSGA-II, distilled

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is an elitist multiobjective
evolutionary algorithm. Each generation it merges parents and offspring into one pool, sorts
that pool into Pareto-dominance fronts with an `O(M N^2)` fast non-dominated sort, and keeps
the best `N` survivors front by front — truncating the one overflowing front by a
parameter-free *crowding distance* that protects sparse regions and the front's extremes.
Parent selection is binary tournament; the DEAP implementation uses direct dominance first,
then crowding distance for nondominating pairs. The method fixes three problems of the earlier
layered-ranking-plus-fitness-sharing approach at once: the `O(M N^3)` sorting cost, the lack of
elitism, and the need to tune a sharing parameter `sigma_share`.

## Problem it solves

Approximate the entire Pareto-optimal front of a multiobjective problem (all objectives
minimized) in a single run, balancing **convergence** to the true front against **diversity**
(uniform spread across the front, including its extremes), cheaply, with elitism, and with no
free parameter governing diversity.

## Three key ideas

1. **Elitism by combine-and-truncate (no external archive).** Form `R_t = P_t ∪ Q_t` of size
   `2N` (parents + offspring), sort it by non-domination, and fill `P_{t+1}` from the best
   fronts down. Because both generations compete in one pool, high-quality parents are not
   discarded merely because a generation ended; when the next front would overflow the remaining
   slots, crowding chooses the representatives that survive.

2. **Fast non-dominated sort, `O(M N^2)`.** In one `O(M N^2)` pass over all pairs, record for
   each `p` its domination count `n_p` (how many dominate it) and dominated set `S_p`.
   Members with `n_p = 0` form front 1. To peel the next front, for each `p` in the current
   front decrement `n_q` for every `q ∈ S_p`; any `q` reaching `n_q = 0` joins the next
   front. The peel only walks stored dominated-set entries, so it is `O(N^2)` after the
   dominance sweep. Total cost is `O(M N^2 + N^2) = O(M N^2)`, down from the naive
   `O(M N^3)`, at the cost of `O(N^2)` storage.

3. **Crowding distance — parameter-free diversity.** Within a front, for each objective `m`
   sort the front, assign the two boundary solutions infinite distance so extremes rank ahead
   of finite-distance interiors during truncation, and give each interior solution the sum over
   objectives of the *normalized neighbor gap*:

   ```
   i.distance += (f_m(i+1) - f_m(i-1)) / (f_m^max - f_m^min)
   ```

   If `f_m^max = f_m^min`, that objective contributes no finite interior gap. DEAP stores the
   average normalized gap by using denominator `M(f_m^max - f_m^min)` for finite updates; this
   common factor preserves the within-front ordering. Boundary solutions are set to infinity
   before any zero-range skip. No `sigma_share`. Cost `O(M N log N)`.

The **crowded-comparison operator** `≺_n` ties rank and crowding together:

```
i ≺_n j   iff   rank_i < rank_j   or   (rank_i = rank_j  and  distance_i > distance_j)
```

Lower (better) front wins; ties go to the less crowded (larger crowding distance) solution.
Environmental selection uses this preference when truncating the first overflowing front; DEAP's
mating tournament uses direct dominance first and the same crowding-distance tie-break.

## Variation operators (real-coded)

- **SBX crossover** (distribution index `eta_c = 20`, probability `p_c = 0.9`). Spread factor
  `beta = |(c2 - c1)/(p2 - p1)|` sampled from `P(beta) = 0.5(eta_c+1)beta^{eta_c}` (`beta<=1`)
  / `0.5(eta_c+1)/beta^{eta_c+2}` (`beta>1`); invert with `u ~ U[0,1]` to
  `beta_q = (2u)^{1/(eta_c+1)}` (`u<=0.5`) else `(1/(2(1-u)))^{1/(eta_c+1)}`; children
  `c1 = 0.5[(1+beta_q)p1 + (1-beta_q)p2]`, `c2 = 0.5[(1-beta_q)p1 + (1+beta_q)p2]`. Children
  land near parents with high probability and their spread is proportional to the parents'
  spread, so the operator self-adapts from exploration to refinement as the population
  contracts.
- **Polynomial mutation** (`eta_m = 20`, `p_m = 1/n`). For `x in [x_L,x_U]`, define
  `delta_L = (x-x_L)/(x_U-x_L)`, `delta_R = (x_U-x)/(x_U-x_L)`, and
  `mut_pow = 1/(eta_m+1)`. If `u < 0.5`, use
  `xy = 1-delta_L`, `val = 2u + (1-2u)xy^{eta_m+1}`,
  `delta_q = val^{mut_pow} - 1`; otherwise use
  `xy = 1-delta_R`, `val = 2(1-u) + 2(u-0.5)xy^{eta_m+1}`,
  `delta_q = 1 - val^{mut_pow}`. Then `x' = x + delta_q(x_U-x_L)` with a final clamp for
  numerical safety. `p_m = 1/n` mutates about one variable per individual regardless of
  dimension.

Both use the bound-respecting forms. DEAP's bounded SBX computes a lower-side `beta`/`alpha`
branch for `c1`, an upper-side `beta`/`alpha` branch for `c2`, and clamps as a numerical guard;
bounded polynomial mutation uses `delta_L` and `delta_R` before the same final clamp.

## Constraint handling (constrained-domination)

No penalty parameter: `i` constrained-dominates `j` if (1) `i` feasible, `j` infeasible; or
(2) both infeasible and `i` has smaller total constraint violation; or (3) both feasible and
`i` dominates `j` normally. Feasible always outranks infeasible; less-violating infeasibles
rank higher. Plugs straight into the same sort; reduces to ordinary dominance when
unconstrained.

## Per-generation cost

`O(M N^2)` (non-dominated sort) + `O(M N log N)` (crowding) + `O(N log N)` (front sort) =
`O(M N^2)`, governed by the non-dominated sort.

## Main loop

```
P_0 <- random population in bounds;  evaluate
fronts <- fast-non-dominated-sort(P_0);  crowding-distance(each front)
for t = 0, 1, 2, ...:
    parents   <- dominance/crowding tournament(P_t, N)
    Q_t       <- SBX(p_c=0.9, eta_c=20) + poly-mutation(p_m=1/n, eta_m=20)(parents); evaluate
    R_t       <- P_t U Q_t                         # size 2N  -> elitism
    fronts    <- fast-non-dominated-sort(R_t)
    P_{t+1}   <- []
    for F in fronts:
        crowding-distance(F)
        if |P_{t+1}| + |F| <= N:  P_{t+1} += F                       # whole front fits
        else:
            sort F by distance desc                                  # first overflowing front
            P_{t+1} += F[: N - |P_{t+1}|];  break                    # keep least crowded
```

## Working code

Filling the one generation-strategy slot of the generational MOEA harness, using DEAP's
`sortNondominated`, `assignCrowdingDist`, `selTournamentDCD`, `cxSimulatedBinaryBounded`, and
`mutPolynomialBounded`:

```python
from copy import deepcopy
import random
from operator import attrgetter

from deap import tools
from deap.tools.emo import assignCrowdingDist


class NSGA2Strategy:
    """One-generation strategy for a real-coded multiobjective EA."""

    def __init__(self, pop_size, n_var, bounds,
                 cx_prob=0.9, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        if pop_size % 4 != 0:
            raise ValueError("selTournamentDCD requires pop_size divisible by 4")
        self.pop_size = pop_size
        self.n_var = n_var
        self.bounds = bounds                                    # (low, up)
        self.cx_prob = cx_prob                                  # p_c = 0.9
        self.cx_eta = cx_eta                                    # eta_c = 20
        self.mut_eta = mut_eta                                  # eta_m = 20
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def _assign_selection_crowding(self, population):
        fronts = tools.sortNondominated(population, len(population),
                                        first_front_only=False)
        for front in fronts:
            assignCrowdingDist(front)

    def clone_and_vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        low, up = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.cx_prob:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=low, up=up)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=low, up=up, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    @staticmethod
    def evaluate_invalid(population, evaluate):
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)

    def generation(self, population, evaluate):
        self._assign_selection_crowding(population)
        parents = tools.selTournamentDCD(population, self.pop_size)
        offspring = self.clone_and_vary(parents)
        self.evaluate_invalid(offspring, evaluate)

        combined = population + offspring                       # elitist 2N pool
        fronts = tools.sortNondominated(combined, self.pop_size,
                                        first_front_only=False)
        for front in fronts:
            assignCrowdingDist(front)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
                continue
            remaining = self.pop_size - len(next_gen)
            front = sorted(front, key=attrgetter("fitness.crowding_dist"), reverse=True)
            next_gen.extend(front[:remaining])
            break
        return next_gen
```

The survival path uses full non-domination fronts and crowding distance; the mating path uses DEAP's
dominance-then-crowding tournament. The bounded crossover and mutation calls are the canonical
real-coded operators used for this method.

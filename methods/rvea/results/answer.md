# RVEA, distilled

RVEA (Reference Vector Guided Evolutionary Algorithm) approximates the Pareto front of a
many-objective problem by attaching one **scalar criterion per reference direction** and
selecting one survivor per direction. The criterion is the **angle-penalized distance (APD)**:
a solution's distance to the ideal point (convergence), inflated by a penalty that grows with
the solution's *angle* to its reference vector (diversity). Because APD is a total order within
each reference vector's subpopulation, selection never goes blind the way Pareto dominance does
once the population is mutually non-dominated — which happens almost immediately as the number
of objectives grows. The penalty ramps from near-zero (converge first) to full strength
(spread later) over the run, and the reference vectors are **adapted to the objective scales**
instead of normalizing the objectives (which would corrupt the angle/length geometry the
criterion depends on).

## Problem it solves

Find a finite, evenly distributed, well-converged approximation to the Pareto front of a
many-objective problem `min F(x) = (f_1(x), ..., f_M(x))`, `M > 3`, in one run — where Pareto
dominance loses selection pressure (nearly everything is non-dominated in high-dimensional
objective space), density-based diversity devices degrade, objective ranges may differ by
orders of magnitude, and the convergence-vs-diversity balance should shift over the run.

## Key idea

- **Reference vectors.** Generate `N` evenly spread directions via the Das-Dennis
  simplex-lattice and map them to the unit hypersphere (`v_i = u_i / ||u_i||`) — directions,
  not simplex weights, since the criterion is angular. Each `v_j` owns a sub-problem.
- **Two orthogonal readings.** Translate objectives by the running ideal estimate
  `z^min` (running per-objective min over the evaluated population) so `f' = f - z^min` sits in
  the first quadrant and the ideal is the origin. Then `||f'||` is the **convergence** reading
  (distance to ideal), and the acute **angle** of `f'` to a reference vector is the **diversity**
  reading
  — bounded in `[0, π/2]`, scale-invariant, and decoupled from `||f'||`.
- **Partition + per-vector elitism.** Associate each individual with the reference vector of
  minimum acute angle (maximum cosine), partitioning the combined parent+offspring pool into
  `N` subpopulations; keep the single minimum-APD individual per non-empty subpopulation.
- **APD = scaled convergence distance.** Multiply, don't add: scaling keeps convergence and
  diversity from tangling and makes the penalty's effect proportional to distance-from-front.
- **Dynamic, self-normalizing penalty.** Near-zero early (pure convergence), growing as
  `(t/t_max)^α` (diversity takes over late), normalized per vector by its nearest-neighbour
  angle `γ_{v_j}` (comparable across dense/sparse niches, without touching the objectives), and
  scaled by `M` (objective-space sparsity grows with `M`).
- **Adapt vectors, not objectives.** Differently-scaled fronts are handled by stretching the
  reference vectors to the observed objective ranges and renormalizing, leaving objective
  values untouched — done only occasionally so the niches don't churn.

## Reference vectors and niche width

- **Generation:** simplex-lattice points `u_i` with entries in `{0/H, ..., H/H}` summing to 1,
  count `N = C(H + M - 1, M - 1)` (two-layer for `M >= 8`), mapped to unit length
  `v_i = u_i / ||u_i||`.
- **Nearest-neighbour angle** `γ_{v_j} = min_{i ≠ j} angle(v_i, v_j)` — the smallest acute
  angle from `v_j` to any other reference vector (computed as the `arccos` of the second-largest
  cosine in `v_j`'s row of `V V^T`). Recomputed whenever the vectors are adapted.

## Angle-penalized distance (APD)

For a translated objective vector `f'_{t,i}` associated with reference vector `v_{t,j}`, with
acute angle `θ_{t,i,j}` between them:

```
d_{t,i,j} = (1 + P(θ_{t,i,j})) · ||f'_{t,i}||

P(θ_{t,i,j}) = M · (t / t_max)^α · θ_{t,i,j} / γ_{v_{t,j}}
```

- `||f'_{t,i}||` — distance to the (translated) ideal point = convergence.
- `θ_{t,i,j} / γ_{v_{t,j}}` — angular deviation normalized by the niche's own width = diversity.
- `(t / t_max)^α` — generation ramp: `≈ 0` early (`P ≈ 0`, `d ≈ ||f'||`, pure convergence),
  `→ 1` late (penalty at full strength, diversity dominates).
- `M` — scales the penalty with the number of objectives.

Per non-empty subpopulation, the survivor is `argmin_i d_{t,i,j}`.

**Vs PBI** (`g^pbi = d_1 + theta · d_2`): APD measures diversity by a bounded, scale-invariant
*angle* rather than PBI's unbounded *Euclidean* off-line distance `d_2` (which is entangled with
distance-from-ideal); APD *scales* the convergence distance instead of *adding* a penalty; and
APD's penalty self-adapts with `t`, `M`, and `γ` so the single `α` does not need re-tuning per
problem, unlike PBI's fixed `theta`.

## Reference vector adaptation

Every `⌈f_r · t_max⌉` generations, fit the reference vectors to the objective scales (leaving
the objectives unchanged):

```
v_{t+1,i} = (v_{0,i} ◦ (z^max_{t+1} - z^min_{t+1})) / ||v_{0,i} ◦ (z^max_{t+1} - z^min_{t+1})||
```

where `◦` is the Hadamard product, `z^max - z^min` is the per-objective range of the current
population, and `v_{0,i}` is the pristine simplex-lattice unit vector (rebuilt from `v_0` each
time so scaling errors don't accumulate). Recompute every `γ_{v_j}` afterward. Done occasionally
(not every generation) because frequent vector motion destabilizes convergence — the niches
keep shifting under the population.

## Final algorithm

```
Input: N reference vectors V_0 (simplex-lattice -> unit sphere); max generations t_max.
Init:  P_0 <- N random individuals; V <- V_0; γ <- nearest-neighbour angles of V.
for t = 0 .. t_max - 1:
    Q  <- offspring-creation(P_t)              # random pairing (no mating selection)
                                               #   + SBX(η_c) + polynomial mutation(η_m)
    R  <- P_t ∪ Q                              # μ + λ elitism
    # --- reference-vector guided selection on R ---
    z^min <- running per-objective min;  f' <- f - z^min   # translate; ||f'|| = dist to ideal
    associate each i with argmax_j cos(f'_i, v_j)          # = min acute angle -> subpopulations
    for each non-empty subpopulation j:
        d_{i,j} = (1 + M·(t/t_max)^α·θ_{i,j}/γ_j) · ||f'_i||
        P_{t+1} <- P_{t+1} ∪ { argmin_i d_{i,j} }          # keep one elitist per vector
    # --- reference-vector adaptation ---
    if t mod ⌈f_r·t_max⌉ == 0:
        z^max <- per-objective max of P_{t+1}; z^min <- per-objective min of P_{t+1}
        V <- normalize(V_0 ◦ (z^max - z^min));  γ <- nearest-neighbour angles of V
Output: P_{t_max}.
```

**Defaults:** `α = 2`, `f_r = 0.1`, SBX `η_c = 30`, `p_c = 1.0`, polynomial mutation `η_m = 20`,
`p_m = 1/n`. Population size set by the simplex-lattice factor `H` and `M`.

**Complexity (per generation):** translation `O(MN)`, partition `O(MN^2)`, APD `O(MN^2)`,
elitism `O(N^2)`, adaptation `O(MN/(f_r t_max))` amortized — overall `O(MN^2)`, with **no
dominance comparison** anywhere.

## Working code

Filling the `select` / `vary` / `survive` / `on_generation` slots of the generation-based
multi-objective EA harness, using the standard DEAP real-coded operators and a clean numpy APD
core (mirroring the canonical pymoo / standalone implementations). The roster is one survivor
per non-empty reference vector; selection is the APD machine; the reference-vector adaptation
runs in `on_generation`.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # cxSimulatedBinaryBounded, mutPolynomialBounded, uniform_reference_points


class CustomMOEA:
    """RVEA: Reference Vector Guided Evolutionary Algorithm. Selects one survivor
    per reference vector by the angle-penalized distance (a scaled distance-to-ideal,
    inflated by a dynamic, niche-normalized angle penalty); adapts the reference
    vectors to the objective scales rather than normalizing the objectives."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=30.0, mut_eta=20.0, mut_prob=None):
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index (RVEA uses 30)
        self.mut_eta = mut_eta                     # polynomial-mutation index (20)
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.alpha = 2.0                           # APD penalty ramp exponent
        self.fr = 0.1                              # reference-vector adaptation frequency
        self.max_gen = 400                         # horizon for the (t/t_max)^alpha ramp;
                                                   #   the driver can overwrite it per problem

        # N evenly spread reference points (Das-Dennis), mapped to the unit sphere.
        p = pop_size - 1 if n_obj == 2 else 12
        V0 = np.asarray(tools.uniform_reference_points(n_obj, p=p), dtype=float)
        self.V0 = self._unit(V0)                   # pristine unit reference vectors
        self.V = self.V0.copy()                    # current (adaptable) reference vectors
        self.pop_size = len(self.V0)
        self.gamma = self._gamma(self.V)           # per-vector nearest-neighbour angle
        self.z_min = None                          # running ideal (per-objective min)
        self._gen = 0

    @staticmethod
    def _unit(V):
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        return V / norms

    def _gamma(self, V):
        # smallest acute angle from each v_j to any OTHER reference vector:
        # arccos of the second-largest cosine in each row (largest = v_j with itself).
        cos = np.clip(V @ V.T, -1.0, 1.0)
        cos_sorted = np.sort(cos, axis=1)
        gamma = np.arccos(cos_sorted[:, -2])
        return np.maximum(gamma, 1e-12)

    def select(self, population, k):
        # No fitness-biased mating selection: each individual already elitist of its subspace.
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        # Random pairing + standard real-coded operators.
        offspring = [deepcopy(ind) for ind in parents]
        random.shuffle(offspring)
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            tools.cxSimulatedBinaryBounded(offspring[i], offspring[i + 1],
                                           eta=self.cx_eta, low=lo, up=hi)  # p_c = 1
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(ind, eta=self.mut_eta, low=lo, up=hi,
                                       indpb=self.mut_prob)
            if ind.fitness.valid:
                del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring          # mu + lambda elitism
        valid = [ind for ind in combined if ind.fitness.valid]
        if len(valid) <= self.pop_size:
            return valid

        F = np.array([ind.fitness.values for ind in valid], dtype=float)
        # Step 1: translate so the ideal is the origin; ||f'|| = distance to ideal.
        z = np.min(F, axis=0)
        self.z_min = z if self.z_min is None else np.minimum(self.z_min, z)
        Fp = F - self.z_min
        dist = np.linalg.norm(Fp, axis=1)
        dist[dist < 1e-12] = 1e-12
        # Step 2: partition by acute angle to the (unit) reference vectors.
        cos = np.clip((Fp / dist[:, None]) @ self.V.T, -1.0, 1.0)
        angle = np.arccos(cos)                     # (n, N)
        assoc = np.argmin(angle, axis=1)
        # Step 3: APD per non-empty subpopulation; keep the min-APD survivor.
        ramp = (self._gen / max(self.max_gen, 1)) ** self.alpha     # (t/t_max)^alpha
        survivors = []
        for j in range(self.pop_size):
            members = np.where(assoc == j)[0]
            if len(members) == 0:
                continue
            theta = angle[members, j]
            penalty = self.n_obj * ramp * (theta / self.gamma[j])
            apd = (1.0 + penalty) * dist[members]
            survivors.append(int(members[int(np.argmin(apd))]))
        # Top up to pop_size from leftovers by smallest distance-to-ideal.
        if len(survivors) < self.pop_size:
            chosen = set(survivors)
            leftover = [i for i in range(len(valid)) if i not in chosen]
            leftover.sort(key=lambda i: dist[i])
            for i in leftover:
                survivors.append(i)
                if len(survivors) >= self.pop_size:
                    break
        return [valid[i] for i in survivors[:self.pop_size]]

    def on_generation(self, gen, population):
        # Remember t for the APD ramp; adapt reference vectors to objective scales.
        self._gen = gen
        period = max(1, int(np.ceil(self.fr * self.max_gen)))
        if gen % period == 0 and population:
            F = np.array([ind.fitness.values for ind in population
                          if ind.fitness.valid], dtype=float)
            if len(F) == 0:
                return
            rng = np.max(F, axis=0) - np.min(F, axis=0)
            rng[rng < 1e-12] = 1e-12
            self.V = self._unit(self.V0 * rng)     # v = v0 ◦ range, renormalized; objectives untouched
            self.gamma = self._gamma(self.V)       # niches moved -> recompute gamma
```

The clean numpy core: reference vectors on the unit sphere, `γ` from the second-largest cosine
per row, translate by the running ideal, partition by `argmin` acute angle, score by
`(1 + M·(t/t_max)^α·θ/γ)·||f'||`, keep the per-vector `argmin`, and adapt the vectors to the
objective ranges every `⌈f_r·t_max⌉` generations.

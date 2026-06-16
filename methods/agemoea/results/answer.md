# AGE-MOEA, distilled

AGE-MOEA (Adaptive GEometry estimation based MOEA) is a real-coded multi-objective
evolutionary algorithm that, every generation, **estimates the Minkowski exponent `p` of the
Pareto front's geometry** from the current best non-dominated front and then trims the critical
front with an **`L_p` survival score that fuses convergence (proximity to the ideal) with
diversity (nearest-neighbor spread)**, both measured in the estimated geometry. It keeps
NSGA-II's generational structure (non-dominated sorting, elitist `(N+N)` survival, binary
tournament, SBX + polynomial mutation) and replaces only the fixed-geometry density metric.

## Problem it solves

A MOEA must approximate a Pareto front with both convergence and even diversity, but both are
distances in objective space and every distance assumes a front *shape*. Existing methods
hard-code one geometry — NSGA-II's crowding distance is an `L_1`/cuboid measure, MOEA/D and
NSGA-III place references on a flat (`p=1`) hyperplane, RVEA measures spread by angle
(spherical) — so each is strong on the shape it assumes and weak on the others, and their
benchmark rankings flip with the front's curvature (convex / linear / concave). Methods that
*fit* the geometry (e.g. GFM-MOEA's Levenberg–Marquardt manifold fit) pay an `O(G'·M^2·(M+N))`
cost that makes continuous per-generation adaptation unattractive. AGE-MOEA keeps the
front-model estimate cheap enough to recompute each generation while retaining the ordinary
sorting and front-distance work of an elitist MOEA.

## Key idea

A normalized front (ideal at origin, scaled to `[0,1]`) sits on the unit `L_p` manifold
`a_1^p + ... + a_M^p = 1`: `p = 1` flat, `p = 2` spherical, `p < 1` convex, `p → ∞` box. The
`L_p` norm with that `p` is the correct ruler for that shape. So:

1. **Normalize** the first (best) front using NSGA-III's procedure: ideal point `z^min` =
   per-axis min, translate to origin; find `M` extreme/corner points (per-axis perpendicular
   distance); fit the hyperplane through them; normalize by its axis intercepts (fallback to
   per-axis max if degenerate).
2. **Estimate `p`** from the single point `C` nearest the diagonal `(1,...,1)` (excluding
   corners), via the manifold equation under the symmetric-center assumption
   `M · mean(C)^p = 1`:
   ```
   p = log(M) / log(1 / mean_i(C_i))   (= log M / (log M − log Σ_i C_i))
   ```
   Clamp: `NaN` or `p <= 0.1 -> p = 1`; if `p > 20`, set `p = 20` to avoid `|a|^p`
   underflow.
   (Recovers `p=1` for a flat front, `p=2` for a sphere.)
3. **Survival score**, in the estimated `L_p`:
   - **Proximity (convergence)** of `A`: `||A||_p` (distance to the origin/ideal); smaller is
     better. Deeper fronts reuse the same normalization and `p`, and the implementation scores
     them by inverse Minkowski distance from `F_d / normalization` to the first-front ideal
     point.
   - **Diversity** of `A`: minimum `L_p` distance to neighbors, `min_{B≠A} ||A − B||_p`.
   - **Fusion + greedy fill on the critical front:** divide each point's pairwise `L_p`
     distances by its own `||A||_p` (so good convergence boosts the gap), keep the `M` corners
     (score `∞`), then greedily add the candidate whose summed distance to its **two** nearest
     already-kept neighbors is largest; record that sum as its score. Trim the critical front
     by descending score.

Binary tournament: lower front rank first, ties broken by higher survival score (the
geometry-aware analogue of NSGA-II's crowded-comparison operator).

## Final algorithm (per generation)

```
P  <- random population of size N
while not done:
    Q  <- variation(select(P))                 # binary tournament, SBX + poly. mutation
    R  <- P ∪ Q                                 # merged pool, |R| = 2N
    {F_1, F_2, ...} <- fast-non-dominated-sort(R)
    # geometry from the best front
    z_min        <- per-axis min of F_1
    F_1          <- (F_1 − z_min)
    extreme      <- M corner points of F_1
    F_1, normz   <- normalize(F_1, extreme)     # hyperplane intercepts (NSGA-III)
    C            <- point of F_1 nearest diagonal (1..1), excluding corners
    p            <- log(M) / log(1 / mean(C))   # NaN or <=0.1 -> 1; >20 -> 20
    score(F_1)   <- survival_score(F_1, p)      # corners ∞; greedy L_p scaled-gap fill
    for d > 1: score(F_d) <- 1 / minkowski(F_d / normz, z_min, p)
    # elitist fill, trim critical front by score
    P <- accept whole fronts in order; keep top-`remaining` of the critical front by score
```

## Default hyperparameters

Population `N` (e.g. 100), SBX crossover with probability `0.9` and distribution index
(`eta_c` ~ 15–20), polynomial mutation with distribution index `eta_m = 20` and
`p_m = 1/n_var`. The variation operators are the standard NSGA-II real-coded pair and are not
the contribution; the survival mechanism is. Numerical guards: `NaN` or `p <= 0.1 -> 1`,
`p > 20 -> 20`, `nn < 1e-8 -> 1`, pairwise `dist < 1e-8 -> 1e-8`, objectives rounded to
about 12 decimals before scoring in the Python implementation.

## Working code (DEAP-style)

Fills the environmental-selection slot of a generational real-coded MOEA. Individuals carry
`fitness.values` (minimized objective tuple) and `fitness.dominates`.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # sortNondominated, cxSimulatedBinaryBounded, mutPolynomialBounded


class AGEMOEA:
    """Adaptive GEometry estimation based MOEA."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                      # (low, up) arrays
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        """Binary tournament: front rank, ties by survival score."""
        fronts = tools.sortNondominated(population, len(population))
        for rank, front in enumerate(fronts):
            for ind in front:
                ind._rank = rank
        out = []
        for _ in range(k):
            a, b = random.sample(population, 2)
            if a._rank < b._rank:
                out.append(deepcopy(a))
            elif b._rank < a._rank:
                out.append(deepcopy(b))
            else:
                sa, sb = getattr(a, "_score", 0.0), getattr(b, "_score", 0.0)
                out.append(deepcopy(a if sa >= sb else b))
        return out

    def vary(self, parents):
        """SBX crossover + polynomial mutation."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1], eta=self.cx_eta, low=lo, up=hi)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring
        F = np.array([ind.fitness.values for ind in combined], dtype=float)
        fronts = tools.sortNondominated(combined, len(combined))
        index_of = {id(ind): i for i, ind in enumerate(combined)}

        score = np.zeros(len(combined))
        for rank, front in enumerate(fronts):
            for ind in front:
                ind._rank = rank

        # geometry from the first (best) front
        f1_idx = [index_of[id(ind)] for ind in fronts[0]]
        ideal = np.min(F[f1_idx], axis=0)
        s1, p, normalization = self._survival_score(F[f1_idx], ideal)
        for local, gi in enumerate(f1_idx):
            score[gi] = s1[local]

        # deeper fronts: proximity-only, same normalization and p
        for rank in range(1, len(fronts)):
            idx = [index_of[id(ind)] for ind in fronts[rank]]
            fr = F[idx] / normalization
            d = self._minkowski_distances(fr, ideal[None, :], p=p).squeeze()
            score[idx] = 1.0 / d

        for ind in combined:
            ind._score = score[index_of[id(ind)]]

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                if remaining <= 0:
                    break
                ranked = sorted(front, key=lambda ind: ind._score, reverse=True)
                next_gen.extend(ranked[:remaining])
                break
        return next_gen

    def _survival_score(self, front, ideal):
        front = np.round(front, 12)
        m, n = front.shape
        scores = np.zeros(m)
        if m < n:
            return scores, 1.0, np.max(front, axis=0)

        front = front - ideal
        extreme = self._find_corners(front)
        front, normalization = self._normalize(front, extreme)

        scores[extreme] = np.inf
        selected = np.zeros(m, dtype=bool)
        selected[extreme] = True

        p = self._estimate_p(front, extreme, n)

        nn = np.linalg.norm(front, p, axis=1)
        nn[nn < 1e-8] = 1.0
        dist = self._pairwise_lp(front, p)
        dist[dist < 1e-8] = 1e-8
        dist = dist / nn[:, None]

        remaining = list(np.where(~selected)[0])
        for _ in range(m - int(selected.sum())):
            sel = np.where(selected)[0]
            D = dist[np.ix_(remaining, sel)]
            if D.shape[1] > 1:
                k = min(2, D.shape[1])
                gap = np.sum(np.partition(D, k - 1, axis=1)[:, :k], axis=1)
            else:
                gap = D[:, 0]
            best = int(np.argmax(gap))
            gi = remaining.pop(best)
            selected[gi] = True
            scores[gi] = gap[best]
        return scores, p, normalization

    @staticmethod
    def _estimate_p(front, extreme, n):
        d = AGEMOEA._point_to_line(front, np.zeros(n), np.ones(n))
        d[extreme] = np.inf
        c = front[int(np.argmin(d))]
        mean_c = float(np.mean(c))
        p = np.log(n) / np.log(1.0 / mean_c)
        if np.isnan(p) or p <= 0.1:
            return 1.0
        return min(p, 20.0)

    @staticmethod
    def _find_corners(front):
        m, n = front.shape
        if m <= n:
            return np.arange(m)
        W = 1e-6 + np.eye(n)
        idx = np.zeros(n, dtype=int)
        taken = np.zeros(m, dtype=bool)
        for i in range(n):
            d = AGEMOEA._point_to_line(front, np.zeros(n), W[i])
            d[taken] = np.inf
            j = int(np.argmin(d))
            idx[i] = j
            taken[j] = True
        return idx

    @staticmethod
    def _point_to_line(P, A, B):
        ba = B - A
        d = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            pa = P[i] - A
            t = np.dot(pa, ba) / np.dot(ba, ba)
            d[i] = np.sum((pa - t * ba) ** 2)
        return d

    @staticmethod
    def _normalize(front, extreme):
        m, n = front.shape
        if len(np.unique(extreme)) != len(extreme):
            norm = np.max(front, axis=0)
            return front / norm, norm
        try:
            beta = np.linalg.solve(front[extreme], np.ones(n))
            if np.any(np.isnan(beta)) or np.any(np.isinf(beta)) or np.any(beta <= 0):
                norm = np.max(front, axis=0)
            else:
                norm = 1.0 / beta
                if np.any(np.isnan(norm)) or np.any(np.isinf(norm)):
                    norm = np.max(front, axis=0)
        except np.linalg.LinAlgError:
            norm = np.max(front, axis=0)
        norm[norm == 0.0] = 1.0
        return front / norm, norm

    @staticmethod
    def _pairwise_lp(front, p):
        m = front.shape[0]
        dist = np.zeros((m, m))
        for i in range(m):
            dist[i] = np.sum(np.abs(front[i] - front) ** p, axis=1) ** (1.0 / p)
        return dist

    @staticmethod
    def _minkowski_distances(A, B, p):
        m1 = A.shape[0]
        m2 = B.shape[0]
        dist = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                dist[i, j] = np.sum(np.abs(A[i] - B[j]) ** p) ** (1.0 / p)
        return dist

    def on_generation(self, gen, population):
        pass
```

## Relation to prior methods

- **NSGA-II** = same generational skeleton, but the critical-front trim uses a fixed `L_1`
  crowding distance with no convergence term; AGE-MOEA replaces it with the `L_p` survival
  score (geometry-adaptive, convergence + diversity).
- **NSGA-III** = AGE-MOEA reuses its extreme-point/hyperplane normalization verbatim, but
  drops the flat (`p=1`) reference-point niching for the estimated-`p` `L_p` metric.
- **GFM-MOEA** = both model the front as an `L_p` manifold, but GFM fits it by Levenberg–
  Marquardt (`O(G'·M^2·(M+N))`, refit every `K` gens) while AGE-MOEA estimates `p` from one
  central point in `O(M)` every generation.

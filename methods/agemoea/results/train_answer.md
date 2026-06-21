A multi-objective evolutionary algorithm has to return a finite set that is both close to the true Pareto front and spread across it. Convergence and diversity are both distances in objective space, and every distance silently assumes a front shape. NSGA-II's crowding distance sums axis-aligned neighbor gaps, which is an L1 cuboid measure that mis-reads density on curved fronts and ignores convergence within a front. MOEA/D and NSGA-III distribute weight or reference points on a flat simplex, so they are calibrated for linear fronts and lose uniformity when the front bends. RVEA uses angle-penalized distances, which assume a spherical geometry. Methods that explicitly fit a parametric front model, such as GFM-MOEA, adapt to the geometry but pay for a Levenberg-Marquardt refit that is too expensive to run every generation. The result is that benchmark rankings flip with the front curvature, because each algorithm hard-codes a different geometry.

The way out is to stop guessing the geometry and instead measure it from the current population each generation. A normalized front, with the ideal point at the origin and scaled to the unit box, can be viewed as an Lp unit manifold a_1^p + ... + a_M^p = 1. The exponent p encodes the shape: p = 1 is a flat simplex, p = 2 is a sphere, p < 1 bulges toward the origin on convex fronts, and p → ∞ approaches a box. Crucially, the Lp norm with the matching p is the natural distance for that shape. If we can estimate p cheaply, we get a geometry-aware ruler for both convergence and diversity without an expensive fitting routine.

The method I propose is AGE-MOEA, the Adaptive GEometry Estimation based Multi-Objective Evolutionary Algorithm. It keeps the standard NSGA-II generational skeleton: fast non-dominated sorting, elitist (N+N) survival, binary tournament mating, and SBX crossover plus polynomial mutation. The only replacement is the survival metric on the critical front. First, AGE-MOEA normalizes the best non-dominated front using NSGA-III's robust procedure. The ideal point is the per-axis minimum. The M extreme or corner points are found by perpendicular distance to each axis direction. A hyperplane is fit through those corners, and the axis intercepts of that hyperplane are used to scale the front into the unit orthant. If the hyperplane is degenerate, it falls back to per-axis maxima.

From the normalized first front, AGE-MOEA estimates the single exponent p. The corners are poor for this purpose because they satisfy the manifold equation for almost any p. The most informative point is the one nearest the diagonal direction (1, ..., 1), the symmetric center of the front. If C is that central point and c̄ is the mean of its coordinates, the symmetric-center assumption M · c̄^p = 1 gives the closed-form estimate p = log(M) / log(1 / c̄). This recovers p = 1 for a flat front and p = 2 for a spherical front, and it is computed in O(M) time with no iterative optimization. The estimate is clamped to the range [0.1, 20] with NaN falling back to p = 1.

With p in hand, AGE-MOEA builds a survival score that fuses convergence and diversity in the estimated geometry. Convergence is measured by proximity to the ideal point, which is the Lp norm ||A||_p of a normalized point. Diversity is measured by Lp distance to already selected neighbors. On the critical front, the M corner points are kept unconditionally because they define the spread. Then the algorithm greedily fills the remaining slots: for each unselected candidate, it sums the Lp distances to its two nearest already-selected neighbors, divides that sum by the candidate's own ||A||_p so that well-converged points are boosted, and selects the candidate with the largest scaled gap. This continues until the front is exhausted. Deeper fronts are scored only by inverse Lp distance to the first-front ideal point, reusing the same normalization and p, so the whole population uses one consistent ruler.

The binary tournament prefers lower non-domination rank and breaks ties by higher survival score, which is the geometry-aware analogue of NSGA-II's crowded-comparison operator. The variation operators remain the standard real-coded pair: SBX crossover with probability 0.9 and distribution index 20, polynomial mutation with distribution index 20 and per-variable probability 1/n_var. The entire novelty sits in the environmental-selection step, and the front geometry is re-estimated every generation at negligible cost.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools


class AGEMOEA:
    """Adaptive GEometry Estimation based MOEA."""

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
                sa = getattr(a, "_score", 0.0)
                sb = getattr(b, "_score", 0.0)
                out.append(deepcopy(a if sa >= sb else b))
        return out

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
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
        F = np.array([ind.fitness.values for ind in combined], dtype=float)
        fronts = tools.sortNondominated(combined, len(combined))
        index_of = {id(ind): i for i, ind in enumerate(combined)}

        score = np.zeros(len(combined))
        for rank, front in enumerate(fronts):
            for ind in front:
                ind._rank = rank

        f1_idx = [index_of[id(ind)] for ind in fronts[0]]
        ideal = np.min(F[f1_idx], axis=0)
        s1, p, normalization = self._survival_score(F[f1_idx], ideal)
        for local, gi in enumerate(f1_idx):
            score[gi] = s1[local]

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
        m1, m2 = A.shape[0], B.shape[0]
        dist = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                dist[i, j] = np.sum(np.abs(A[i] - B[j]) ** p) ** (1.0 / p)
        return dist

    def on_generation(self, gen, population):
        pass
```

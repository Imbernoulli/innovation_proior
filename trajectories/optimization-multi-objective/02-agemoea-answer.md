**Problem.** Both the default (NSGA-II) and decomposition hard-code a front shape into their diversity ruler:
crowding distance is an L1 cuboid that mis-reads density on curved fronts, and decomposition's uniform weights
are calibrated for a flat simplex. Decomposition's DTLZ2 spread (0.846) and DTLZ1 spread (4.41) confirmed the
penalty. Convergence and diversity are distances, and every distance assumes a geometry.

**Key idea.** Stop hard-coding the geometry: measure the Pareto front's Minkowski exponent `p` from the
current population each generation, and use the matching Lp norm as the diversity ruler. A normalized front is
the Lp unit manifold `sum_j a_j^p = 1` — p=1 flat, p=2 spherical, p<1 convex, p→∞ box. Keep NSGA-II's elitist
non-dominated-sort skeleton, but trim the critical front by Lp nearest-neighbor diversity instead of the fixed
cuboid crowding distance.

**Why (and what this harness leaves out).** This edit surface ships a *stripped* adaptive-geometry method:
`p` is estimated by a coarse **binary search** on the median of a min/max-normalized front (not the canonical
closed form over the diagonal-central point), normalization is per-axis min/max (not the NSGA-III hyperplane
fit), and the survival score is **diversity only** — the single nearest-neighbor Lp distance — with no
proximity-to-ideal (convergence) term and no greedy corner-seeded fill. So it should repair spread on the
curved 3-objective fronts but leave DTLZ2 convergence (HV/IGD) roughly where decomposition left it.

**Hyperparameters.** Geometry exponent `p` clamped to [0.1, 20], binary-searched (50 iters) on the median
point; survival = Lp nearest-neighbor distance on the critical front; rank-only binary tournament for mating;
SBX `eta_c = 20` at p=0.9, polynomial mutation `eta_m = 20`, `p_m = 1/n_var`.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 2: AGE-MOEA
class CustomMOEA:
    """AGE-MOEA: Adaptive Geometry Estimation based MOEA."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def _estimate_geometry(self, front_values):
        """Estimate the geometry parameter p of the Pareto front."""
        if len(front_values) < 2 or self.n_obj < 2:
            return 1.0

        F = np.array(front_values)

        # Normalize objectives
        z_min = np.min(F, axis=0)
        z_max = np.max(F, axis=0)
        scale = z_max - z_min
        scale[scale < 1e-12] = 1.0
        F_norm = (F - z_min) / scale

        # Find extreme points (closest to axes)
        extremes = []
        for m in range(self.n_obj):
            idx = np.argmin(F_norm[:, m])
            extremes.append(F_norm[idx])

        if len(extremes) < 2:
            return 1.0

        # Use the median point on the front to estimate p
        median_idx = len(F_norm) // 2
        median_point = np.sort(F_norm, axis=0)[median_idx]
        median_point = np.maximum(median_point, 1e-8)

        # Binary search for p
        p_low, p_high = 0.1, 20.0
        for _ in range(50):
            p_mid = (p_low + p_high) / 2
            lp_val = np.sum(median_point ** p_mid)
            if lp_val > 1.0:
                p_low = p_mid
            else:
                p_high = p_mid
        p = (p_low + p_high) / 2
        return max(0.1, min(p, 20.0))

    def _survival_score(self, front_values, p):
        """Compute survival score based on Lp-distance-based crowding."""
        F = np.array(front_values)
        n = len(F)
        if n <= 2:
            return np.full(n, float('inf'))

        # Normalize
        z_min = np.min(F, axis=0)
        z_max = np.max(F, axis=0)
        scale = z_max - z_min
        scale[scale < 1e-12] = 1.0
        F_norm = (F - z_min) / scale

        # Compute pairwise Lp-distances
        scores = np.zeros(n)
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                diff = np.abs(F_norm[i] - F_norm[j])
                lp_dist = np.sum(diff ** p) ** (1.0 / p)
                dists.append(lp_dist)
            dists.sort()
            if dists:
                scores[i] = dists[0]
            else:
                scores[i] = 0.0

        return scores

    def select(self, population, k):
        """Binary tournament selection based on non-domination rank."""
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for rank, front in enumerate(fronts):
            for ind in front:
                ind.fitness.crowding_dist = 0.0  # reset
                ind._rank = rank
        selected = []
        for _ in range(k):
            i1, i2 = random.sample(range(len(population)), 2)
            a, b = population[i1], population[i2]
            if a._rank < b._rank:
                selected.append(deepcopy(a))
            elif b._rank < a._rank:
                selected.append(deepcopy(b))
            else:
                selected.append(deepcopy(random.choice([a, b])))
        return selected

    def vary(self, parents):
        """SBX crossover + polynomial mutation."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < 1.0:
                tools.mutPolynomialBounded(
                    ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
                )
                del ind.fitness.values

        return offspring

    def survive(self, population, offspring):
        """AGE-MOEA survival: adaptive geometry-based selection."""
        combined = population + offspring

        # Non-dominated sorting
        fronts = tools.sortNondominated(combined, len(combined), first_front_only=False)

        next_gen = []
        for front_idx, front in enumerate(fronts):
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                if remaining <= 0:
                    break

                # Estimate geometry from the first front
                first_front_values = [ind.fitness.values for ind in fronts[0]]
                p = self._estimate_geometry(first_front_values)

                # Compute survival scores for the critical front
                front_values = [ind.fitness.values for ind in front]
                scores = self._survival_score(front_values, p)

                # Select individuals with highest diversity scores
                sorted_indices = np.argsort(-scores)  # descending
                for idx in sorted_indices[:remaining]:
                    next_gen.append(front[idx])
                break

        return next_gen

    def on_generation(self, gen, population):
        pass
```

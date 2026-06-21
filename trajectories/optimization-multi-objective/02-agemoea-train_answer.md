Decomposition came in at the floor, and it failed for exactly the reason I planted it there. On the 2-objective ZDTs it converged respectably (ZDT1 HV mean 0.8627) and was the fastest family in the suite, but on the 3-objective spherical front DTLZ2 its HV was only 2.737 with IGD 0.0761 — an order of magnitude worse than the 2-objective IGDs — and its spread numbers were an indictment: DTLZ2 spread 0.846, DTLZ1 spread mean 4.41 swinging from 0.61 to 8.49 across seeds. The plain Tchebycheff lattice, with no boundary-intersection geometry, simply cannot lay points uniformly on a curved 3-objective front, and the uncapped neighbor takeovers let a few good children overwrite swaths of subproblems and clump the population. The deeper diagnosis is that decomposition baked in *flat* — uniform weights on a simplex face — and paid for it on the sphere. Convergence and diversity are both distances in objective space, and every distance secretly assumes a shape: the same pair of points reads as crowded under one geometry and well-spread under another. The default's crowding distance bakes in a *different* shape, an $L_1$ cuboid (sum of per-axis normalized neighbor gaps), which also mis-reads a curved front. So the rung I want stops hard-coding a geometry and instead measures the front's shape from the population and uses a matching ruler.

I propose **AGE-MOEA**: keep NSGA-II's cheap elitist non-dominated-sort skeleton, but replace both the fixed weights and the $L_1$ cuboid with a distance whose shape is read off the data each generation. The clean handle on "front geometry" is the Minkowski exponent. Translate so the ideal sits at the origin and scale so the front spans $[0,1]$ per axis; then the family of front shapes is the unit sphere of the $L_p$ norm, the points with $\sum_j a_j^p = 1$ in the positive orthant. The exponent $p$ tells the whole story: at $p = 1$ it is the flat simplex face (DTLZ1's true linear front), at $p = 2$ the Euclidean sphere (DTLZ2), at $p < 1$ it bulges convex toward the origin (ZDT1), and as $p \to \infty$ it becomes the box. The whole zoo of shapes is one scalar — and the $L_p$ norm with that same $p$ is the natural distance on a front of that shape. If I can read off $p$, I get the right ruler for free.

So step one is to estimate $p$ from the current front, per generation. I must be precise about what this edit surface implements, because it is a stripped version and the gaps it leaves are what I will read in the numbers next. The full version of the idea normalizes via NSGA-III's robust extreme-point hyperplane and then estimates $p$ in closed form from the single point nearest the diagonal — $M\bar{c}^{\,p} = 1$ collapses to $p = \log(M)/\log(1/\text{mean}(C))$, recovering $p=1$ on a flat front and $p=2$ on a sphere from one anchor with no iterative fit. This task's edit surface does *neither*. It normalizes crudely by per-axis min/max, $F_{\text{norm}} = (F - z_{\min})/(z_{\max} - z_{\min})$, and it estimates $p$ not by the closed form but by a *binary search*: take the median point of the normalized first front and bisect $p$ in $[0.1, 20]$ until $\sum_j (\text{median}_j)^p$ crosses 1. That is coarser and more fragile — the median is a noisier anchor than the diagonal-central point, and min/max normalization is more sensitive to a single outlier than the hyperplane fit — so I should expect it to track the front shape only roughly, and I keep the $[0.1, 20]$ clamp because $\lvert a\rvert^p$ underflows or explodes outside that band.

The survival metric carries the load-bearing divergence from the canonical idea, the one I expect to determine how far this rung climbs. The full adaptive-geometry method fuses *two* qualities into one survival score: convergence (proximity to the ideal, $1/\lVert A\rVert_p$, smaller norm = better converged) and diversity (a greedy nearest-selected-neighbor $L_p$ gap fill seeded by the kept corners), dividing each candidate's neighbor gaps by its own $\lVert A\rVert_p$ so a point that is both isolated *and* well-converged is rewarded twice. This task's edit surface implements *only the diversity half*. For each point on the critical front it computes the $L_p$ distance to every other point and keeps the *single nearest-neighbor distance* as the score — pure spread, in the estimated $L_p$ geometry, with no proximity-to-ideal term, no corner seeding, and no greedy max-min fill. So what ships is NSGA-II's elitist skeleton with the $L_1$ cuboid crowding distance swapped for an $L_p$ nearest-neighbor diversity whose $p$ is read (crudely) from the front each generation. That is a real improvement over the fixed cuboid on *diversity* — the ruler now bends to the front shape — but it does nothing for *convergence within a front*, which is precisely the quality plain crowding distance also ignored. I am honest that this rung fixes the spread problem and not the convergence problem.

End to end, the environmental selection runs exactly as NSGA-II does until the critical front. Merge parents and offspring into the $2N$ pool, non-dominated-sort into $F_1, F_2, \dots$, fill the next generation front by front while whole fronts fit. For the front that overflows: estimate $p$ from the *first* front's values (binary search on the median of the min/max-normalized first front), compute the $L_p$ nearest-neighbor diversity score for every point on the critical front, sort that front by score descending, and keep the highest-scoring points until full. Parent selection is a binary tournament on non-domination *rank only* — each individual gets its front index `_rank`, crowding is reset to zero, and the lower rank wins (ties by coin flip) — so the geometry-awareness lives entirely in `survive`, not in `select`. Variation stays the shared real-coded pair (SBX $\eta_c = 20$ at probability 0.9, polynomial mutation $\eta_m = 20$, $p_m = 1/n$), so the only difference from the other rungs is the survival metric. I expect this to repair spread on the curved 3-objective fronts but leave DTLZ2 convergence (HV/IGD) roughly where decomposition left it — a clear spread win, a convergence wash, leaving DTLZ2 HV as the open wound the next rung must close.

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

**Problem.** Every prior rung optimizes a *surrogate* — dominance plus a density proxy (NSGA-II crowding,
SPEA2 k-NN), or coverage of a fixed reference grid (NSGA-III) — and is then graded by the headline metric,
**hypervolume**. SPEA2 won on aggregate by dominating spread and IGD, but its DTLZ2 HV (2.781) actually trailed
NSGA-III (2.790): no rung leads the very metric the task scores.

**Key idea.** Make selection aim directly at hypervolume. The fixed-size set that maximizes dominated
hypervolume is automatically converged *and* well-spread (and strictly Pareto-compliant). Keep the
non-dominated-sorting skeleton (respect dominance first), and to shrink the pool remove, from the *worst*
front, the **least hypervolume contributor** — the point whose deletion costs the least dominated volume.
That one quantity does crowd-thinning (near-duplicates contribute ~0), convergence (poorly-converged points
contribute little), and boundary preservation (extremes carve a long slab toward the reference, never the
minimum) — and it is the scored indicator itself, not a surrogate.

**Why it is the endpoint.** It turns the grading measure into the selection rule, so it should *lead* on HV
where SPEA2 only tied. Respecting dominance first stops a dominated point in empty space from outranking a
non-dominated one; when the worst front is the first front (everything non-dominated), the contribution is
the whole selection — exactly where crowding and grid coverage fail. Exact 2D (`O(n log n)`) and 3D
(`O(n^2)`) hypervolume — reproduced from the harness's own grader so the optimized quantity matches the
scored one — make it affordable, at the cost of being the slowest rung.

**Reference point.** The one parameter: set adaptively each generation just beyond the pool's nadir,
`ref = nadir + 0.1·(nadir - ideal)`, so boundary points keep a finite positive contribution and the scale
tracks the front (no per-problem tuning).

**Hyperparameters.** Reference margin 0.1; NSGA-II dominance-crowding tournament mating; SBX `eta_c = 20` at
p=0.9; polynomial mutation `eta_m = 20`, `p_m = 1/n_var`.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — finale: SMS-EMOA
class CustomMOEA:
    """SMS-EMOA: S-metric (hypervolume) selection EMOA."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        # Reference point for the dominated-hypervolume indicator, set adaptively
        # each generation from the current pool (nadir + margin).
        self.ref_margin = 0.1
        self._ref_point = None

    # ---- exact hypervolume (matches the harness's 2D/3D computation) ----
    def _hv_2d(self, points, ref):
        pts = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
        if len(pts) == 0:
            return 0.0
        pts = pts[pts[:, 0].argsort()]
        nd = [pts[0]]
        for p in pts[1:]:
            if p[1] < nd[-1][1]:
                nd.append(p)
        nd = np.array(nd)
        hv = 0.0
        prev_y = ref[1]
        for p in nd:
            hv += (ref[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]
        return hv

    def _hv_3d(self, points, ref):
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 2])]
        hv = 0.0
        active_2d = []
        for i in range(len(pts)):
            active_2d.append(pts[i, :2])
            z_lo = pts[i, 2]
            z_hi = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
            dz = z_hi - z_lo
            if dz > 0:
                hv += self._hv_2d(np.array(active_2d), ref[:2]) * dz
        return hv

    def _hypervolume(self, points, ref):
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return 0.0
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
        if self.n_obj == 2:
            return self._hv_2d(points, ref)
        if self.n_obj == 3:
            return self._hv_3d(points, ref)
        return 0.0

    def _least_contributor(self, front, ref):
        """Index of the front member whose removal shrinks the dominated
        hypervolume the least (the S-metric least contributor)."""
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        n = len(F)
        total = self._hypervolume(F, ref)
        best_idx, best_contrib = 0, float("inf")
        for i in range(n):
            sub = np.delete(F, i, axis=0)
            contrib = total - self._hypervolume(sub, ref)
            if contrib < best_contrib:
                best_contrib = contrib
                best_idx = i
        return best_idx

    def select(self, population, k):
        """Binary tournament on non-domination rank + crowding distance."""
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

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
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
            )
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        """SMS-EMOA survival: non-dominated sorting, then remove the least
        hypervolume contributor from the worst front until pop_size remain."""
        combined = [ind for ind in population + offspring if ind.fitness.valid]
        if len(combined) <= self.pop_size:
            return combined

        # Adaptive reference point: nadir of the current pool plus a margin so
        # boundary solutions retain a finite, positive hypervolume contribution.
        all_vals = np.array([ind.fitness.values for ind in combined], dtype=float)
        nadir = np.max(all_vals, axis=0)
        ideal = np.min(all_vals, axis=0)
        span = nadir - ideal
        span[span < 1e-12] = 1.0
        self._ref_point = nadir + self.ref_margin * span

        survivors = list(combined)
        while len(survivors) > self.pop_size:
            fronts = tools.sortNondominated(survivors, len(survivors), first_front_only=False)
            worst = fronts[-1]
            if len(worst) == 1:
                # A singleton worst front: drop it directly.
                drop = worst[0]
            else:
                idx = self._least_contributor(worst, self._ref_point)
                drop = worst[idx]
            survivors.remove(drop)
        return survivors

    def on_generation(self, gen, population):
        pass
```

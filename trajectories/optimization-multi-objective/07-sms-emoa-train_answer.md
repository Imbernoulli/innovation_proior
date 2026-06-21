SPEA2 closed the ladder's main gap: a step-change in spread (ZDT1 0.149 against NSGA-III's 0.287, DTLZ2 0.106 against 0.621, DTLZ1 0.138 against 0.648) achieved *without* a convergence regression (DTLZ2 HV 2.781 with the best IGD of any rung, 0.0498), exactly the clean spread win I predicted, because the k-NN truncation optimizes real inter-point distances — the quantity the Spread metric actually scores. But notice what SPEA2 still does *not* optimize: the headline metric, hypervolume. Its DTLZ2 HV (2.781) is actually a hair *below* NSGA-III's (2.790) and RVEA's (2.789); it won on aggregate by dominating spread and IGD, not by leading HV. That is the residual every rung shares — each optimizes a *surrogate* (dominance plus a density proxy, or coverage of a reference grid) and is then graded by hypervolume, aiming at one target and scored against another. The natural next move is to make selection aim *directly* at the measure it is judged by.

I propose **SMS-EMOA**: S-metric (hypervolume) selection. Hypervolume is the field's strictly Pareto-compliant quality measure — if a set dominates another, its hypervolume is strictly larger — and the fixed-size set that maximizes it is automatically *both* converged (the points must hug the front) and well-spread (they must not pile up), concentrating resolution where the front is more strongly curved. So hypervolume already fuses the two qualities the whole ladder has been balancing with separate mechanisms. The exact target — among all size-$N$ subsets of the combined pool, keep the one of maximum HV — is combinatorial and unaffordable, so the standard relaxation is greedy *removal*: shrink the pool one point at a time, and at each step remove the point whose deletion costs the least dominated hypervolume. The quantity I need per point is its **hypervolume contribution**,

$$\Delta_{\text{HV}}(p) = \text{HV}(P) - \text{HV}(P \setminus \{p\}),$$

the volume dominated *only* by that point. Removing the least contributor is the unification SPEA2's two-part fitness only approximated. A near-duplicate on the front has a tiny unique contribution (its partner already dominates almost the same volume), so it is dropped — crowd-thinning. A poorly-converged point behind the front dominates little volume relative to the points ahead of it, so its contribution is small and it is pruned in favor of better-converged points — convergence pressure. And a boundary/extreme point dominates a long thin slab of volume out toward the reference that no interior point touches, so its contribution is large and it is *never* the least contributor — boundary preservation, automatic from the volume geometry, the very property SPEA2 had to engineer with a lexicographic tie-break. All three pressures from one quantity, and unlike SPEA2's k-NN density this *is* the scored indicator itself.

One guard is where a naive hypervolume-only rule goes wrong. A dominated point sitting alone in an empty corner of objective space can dominate a chunk of volume nobody else reaches, so its raw contribution can exceed that of a crowded but *non-dominated* first-front point. Ranking purely by contribution would keep the dominated straggler and cut the better point — a violation of the dominance ordering hypervolume is supposed to respect. So I honor dominance *first* and use the contribution only to choose *within* a dominance class. The clean way reuses the non-dominated-sorting skeleton the whole ladder already runs: sort the pool into fronts, and when I must remove a point, remove it from the *worst* (highest-index) front, letting the least-contributor rule pick which member of that worst front to drop. Dominance handles the coarse convergence ordering exactly as in NSGA-II and SPEA2; hypervolume handles the fine within-front choice — and when the worst front *is* the first front (the regime where everything is mutually non-dominated and crowding distance went blind), the contribution becomes the entire selection, which is precisely where direct hypervolume should beat both crowding and grid coverage. I re-sort each iteration because removing one point can promote previously-dominated points to a better front, keeping the dominance ordering honest as points leave.

The harness already exposes everything I need: `tools.sortNondominated` for the front skeleton, and the *same exact* 2D/3D hypervolume routines the grader uses are reproducible inside `survive` — a 2D non-dominated sweep in $O(n\log n)$, and a 3D plane-sweep up the $z$-axis maintaining the 2D hypervolume of the swept points times the slab thickness, $O(n^2)$. I reproduce those exactly so the contribution I optimize *is* the indicator the task scores, not an approximation — that fidelity is the whole point of the rung. The one genuine parameter is the reference point, and it controls the boundary behavior I am relying on. If the reference sits *on* the nadir, an extreme point's slab has zero thickness and its contribution collapses to zero — it would become the least contributor and be wrongly discarded, destroying the spread. So the reference must sit strictly beyond the nadir by a margin, but not so far that contributions are dominated by distance-to-reference and the within-front discrimination flattens. I set it adaptively each generation,

$$\text{ref} = \text{nadir} + 0.1\cdot(\text{nadir} - \text{ideal}),$$

the pool's per-objective maximum pushed out by a tenth of the front's span, so boundary points keep a finite positive contribution and the scale tracks the front with no per-problem tuning — which matters because ZDT's $[0,1]$-scaled fronts and the DTLZ fronts have different magnitudes.

The rest I keep standard so the difference from the prior rungs is the survival rule, not the operators: `select` is the NSGA-II dominance-crowding binary tournament (good, spread-out parents, cheap, with the heavy lifting downstream), `vary` is the shared SBX ($\eta_c = 20$, $p_c = 0.9$) plus polynomial mutation ($\eta_m = 20$, $p_m = 1/n$). The whole novelty is `survive`: compute the adaptive reference, then loop — non-dominated-sort the current survivors, take the worst front, drop it directly if it is a singleton, otherwise compute every member's hypervolume contribution and remove the least contributor — until exactly `pop_size` remain. I verified the contribution logic standalone against the harness's own 2D/3D hypervolume: on a uniform 2D front the redundant interior point is the least contributor; adding a near-duplicate makes *it* the least contributor (contribution $\approx 0.0009$); the axis extremes are never selected for removal — exactly the crowd-thin-and-preserve-extremes behavior the derivation predicts. The claim, the one that justifies the endpoint, is that optimizing the scored indicator directly should *lead on HV* where SPEA2 only tied: DTLZ2 HV should reach or exceed the best prior rung's 2.790, and on the ZDTs HV should sit at or above SPEA2's 0.8682 and 1.3243. On spread I expect it competitive with SPEA2 — hypervolume's optimal distribution is well-spread and boundary-preserving — though not necessarily to beat SPEA2's 0.149/0.106, since SMS-EMOA optimizes spread only as a by-product of volume; if spread comes in materially worse while HV leads, that is the honest trade on an HV-primary task. The one cost is wall-clock — the per-removal contributions over a $2N\to N$ reduction make this the most expensive rung by a wide margin — but cost is not a scored metric here.

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

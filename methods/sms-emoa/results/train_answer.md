A multi-objective minimization problem asks for decisions whose objective vectors $f(x) = (f_1(x), \dots, f_m(x))$ cannot be ordered by a single scalar loss, so the deliverable is not a point but a finite approximation to the Pareto front: a set that both moves close to the true front and covers its trade-off surface without collapsing onto one region or losing the extremes. The hard part of building such an approximation inside an elitist evolutionary algorithm is that survival has two jobs that pull against each other. It must respect Pareto dominance, because a dominated solution should never outlive a nondominated one, and it must also impose pressure among solutions that dominance cannot separate — and on a single nondominated front, dominance is silent exactly where most of the selection happens. The established elitist EMOAs answer the second job with density surrogates: NSGA-II keeps the points with the largest crowding distance, an axis-aligned sum of normalized neighbour gaps that protects boundary points; SPEA2 mixes a strength count with a $k$-th-nearest-neighbour density term and an archive truncation pass. Both are cheap and robust, but they are local geometric proxies, not the quantity by which the final set is actually judged. IBEA makes indicators explicit but only through pairwise binary indicators with an exponential scaling constant, so its fitness is a parameterized pairwise score rather than a set-level decision. Adaptive archiving ties retention to dominated-volume arguments but tends to shrink the active population or split the archive off from the variation pool. The recurring gap is that the within-front survival pressure disagrees with the set-quality indicator most studies ultimately report — the dominated hypervolume, or S-metric, the Lebesgue measure $S(A, r) = \lambda(\{ y : \exists\, a \in A,\ a \le y \le r \})$ of the region dominated by the set $A$ up to a reference point $r$ worse than all points of interest. The S-metric rewards convergence, because pushing a point toward the front enlarges its dominated box, and it rewards coverage, because spreading points stops the same dominated region from being counted twice. The reason nobody simply selects on it is cost: exact hypervolume is cheap in two objectives and manageable in three, but a survival rule that re-evaluates large subsets every generation would be far more expensive than a density pass.

I propose SMS-EMOA, the S-metric selection evolutionary multi-objective algorithm, which keeps the dependable dominance ranking as the primary rule and makes the within-front pressure agree with the hypervolume indicator, without ever solving an intractable subset problem. The defining move is to fix the rhythm rather than the metric. The naive target — after generating a whole offspring population, retain the size-$\mu$ subset of $\mu + \lambda$ with the largest $S$ — is an exponential subset search. Instead the algorithm runs steady-state: it creates exactly one evaluated child, forms a pool of size $\mu + 1$, and deletes exactly one member. The set-level objective then collapses to a one-delete question, and the quantity that governs it is the exclusive hypervolume contribution of a candidate $s$ inside a set $R$,
$$\Delta S(s, R) = S(R) - S(R \setminus \{s\}),$$
the volume that would be lost by removing $s$. This has exactly the sign structure I want: a large positive $\Delta S$ means $s$ uniquely owns a slab of dominated space, a near-zero $\Delta S$ means $s$ is redundant, and the right point to discard is the one with the minimum contribution. The subtraction order is load-bearing — written the other way, useful points would look negative and the most valuable point would be selected for deletion. Dominance still comes first: I partition the $\mu + 1$ pool into nondominated-sorting fronts $R_1, \dots, R_v$ and make only the worst front $R_v$ eligible for deletion, so any point on a worse level is removed before any better-front point. If $R_v$ holds one point the choice is forced; if it holds several mutually nondominated points, I delete the smallest hypervolume contributor among them. Rank first, volume-contribution second — the indicator never overrides dominance.

What makes this practical is that the contribution can be computed exactly and cheaply in the dimensions the algorithm targets, with the reference-point dependence handled carefully. In two objectives, a nondominated front sorted by increasing $f_1$ is automatically sorted by decreasing $f_2$, and an interior point $s_i$ owns a single exclusive rectangle that starts at its own coordinates, extends in $f_1$ to the next point, and in $f_2$ to the previous point, so
$$\Delta S(s_i, R_v) = \big(f_1(s_{i+1}) - f_1(s_i)\big)\big(f_2(s_{i-1}) - f_2(s_i)\big),$$
both factors nonnegative under that ordering. A small factor means the point is crowded against a neighbour and is a natural deletion candidate; swapping either neighbour — using the previous point in $f_1$ or the next in $f_2$ — would turn the exclusive rectangle into the wrong area. The two extremes are special because their boxes reach all the way to $r$; in 2D I avoid letting the reference point decide the endpoints by protecting both extremes and computing finite contributions only for interior points, falling back to an ordinary finite-reference hypervolume contribution only in the degenerate case where the front has no interior point so the code still returns a deletion. In three objectives I cannot protect every boundary point, because many points can carry one worst coordinate and protecting all of them would turn "boundary" into a population-size trap, so I need a concrete reference and choose it dynamically as the coordinatewise worst vector of the current front plus an offset, $r = \text{nadir} + 1.0$. The $+1.0$ gives boundary boxes positive thickness without introducing a scale-dependent span multiplier, and whenever I need exact contributions beyond the simple 2D interior formula I just compute $S(R, r) - S(R \setminus \{s\}, r)$ against that reference. I also keep one optional variant for the case where the worst front $R_v$ is itself dominated by something better: rather than spend hypervolume computation on already-dominated material, delete the point of $R_v$ dominated by the largest number of pool members; when the whole pool is nondominated this count is zero for everyone and the rule reverts to the hypervolume contribution. The full algorithm is then small — initialize a fixed population, then repeatedly generate one child, evaluate it, append it, nondominated-sort the $\mu + 1$ pool, and delete one member of the worst front by minimum contribution (or, in the variant, by maximum domination count) — and because each step removes only one member the population size stays fixed and the retained relevant front's S-metric cannot fall by accidentally discarding a better contributor. The variation operators are left ordinary; SBX recombination and polynomial mutation, or any evaluated-offspring generator, suffice, because the entire novelty lives in a reduction hook that reads only objective vectors.

```python
import numpy as np


class SMSEMOA:
    def __init__(self, pop_size, make_child=None, selection_mode="least_contributor", ref_offset=1.0):
        if selection_mode not in {"least_contributor", "domination_count"}:
            raise ValueError("selection_mode must be 'least_contributor' or 'domination_count'")
        self.pop_size = pop_size
        self.make_child = make_child
        self.selection_mode = selection_mode
        self.ref_offset = float(ref_offset)

    def step(self, population):
        if self.make_child is None:
            raise ValueError("step() needs a make_child callable")
        child = self.make_child(population)
        return self.reduce(population + [child])

    def reduce(self, pool):
        if len(pool) != self.pop_size + 1:
            raise ValueError("SMS-EMOA reduction expects a mu+1 pool")
        F = np.asarray([ind.fitness.values for ind in pool], dtype=float)
        drop = self.discard_index(F)
        return [ind for i, ind in enumerate(pool) if i != drop]

    def discard_index(self, F):
        F = np.asarray(F, dtype=float)
        fronts = self._nondominated_fronts(F)
        worst = fronts[-1]
        if len(worst) == 1:
            return worst[0]
        if self.selection_mode == "domination_count" and len(fronts) > 1:
            counts = [sum(self._dominates(F[j], F[i]) for j in range(len(F)) if j != i) for i in worst]
            return worst[int(np.argmax(counts))]
        return self._least_hv_contributor(F, worst)

    @staticmethod
    def _dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def _nondominated_fronts(self, F):
        n = len(F)
        dominates = [[] for _ in range(n)]
        dominated_by = np.zeros(n, dtype=int)
        first = []
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(F[p], F[q]):
                    dominates[p].append(q)
                elif self._dominates(F[q], F[p]):
                    dominated_by[p] += 1
            if dominated_by[p] == 0:
                first.append(p)
        fronts = [first]
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_by[q] -= 1
                    if dominated_by[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    def _least_hv_contributor(self, F, front):
        front = np.asarray(front, dtype=int)
        points = F[front]
        if points.shape[1] == 2:
            idx = self._least_2d_contributor_with_extremes(points)
            if idx is not None:
                return front[idx]
        ref = np.max(points, axis=0) + self.ref_offset
        contrib = self._hv_contributions(points, ref)
        return front[int(np.argmin(contrib))]

    @staticmethod
    def _least_2d_contributor_with_extremes(points):
        if len(points) <= 2:
            return None
        order = np.argsort(points[:, 0], kind="mergesort")
        sorted_points = points[order]
        contrib = np.full(len(points), np.inf)
        for i in range(1, len(sorted_points) - 1):
            dx = sorted_points[i + 1, 0] - sorted_points[i, 0]
            dy = sorted_points[i - 1, 1] - sorted_points[i, 1]
            contrib[i] = max(dx, 0.0) * max(dy, 0.0)
        return int(order[int(np.argmin(contrib))])

    def _hv_contributions(self, points, ref):
        total = self._hypervolume(points, ref)
        return np.array([
            total - self._hypervolume(np.delete(points, i, axis=0), ref)
            for i in range(len(points))
        ])

    def _hypervolume(self, points, ref):
        points = np.asarray(points, dtype=float)
        ref = np.asarray(ref, dtype=float)
        points = points[np.all(points < ref, axis=1)]
        if len(points) == 0:
            return 0.0
        if points.shape[1] == 2:
            return self._hv2d(points, ref)
        if points.shape[1] == 3:
            return self._hv3d(points, ref)
        raise NotImplementedError("exact hypervolume contribution is implemented here for 2D and 3D")

    @staticmethod
    def _hv2d(points, ref):
        order = np.lexsort((points[:, 1], points[:, 0]))
        pts = points[order]
        stair = []
        best_y = np.inf
        for p in pts:
            if p[1] < best_y:
                stair.append(p)
                best_y = p[1]
        hv = 0.0
        prev_y = ref[1]
        for x, y in stair:
            hv += max(ref[0] - x, 0.0) * max(prev_y - y, 0.0)
            prev_y = min(prev_y, y)
        return hv

    def _hv3d(self, points, ref):
        pts = points[np.argsort(points[:, 2], kind="mergesort")]
        hv = 0.0
        active = []
        for i, p in enumerate(pts):
            active.append(p[:2])
            z_hi = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
            dz = z_hi - p[2]
            if dz > 0:
                hv += self._hv2d(np.asarray(active), ref[:2]) * dz
        return hv
```

I start with the part that is hard to ignore: dominance gives me a partial order, but it goes silent exactly where most of the selection work happens. Once two solutions are on the same nondominated front, neither dominates the other, and I still have to decide which one deserves a place in a finite population. NSGA-II answers with crowding distance. SPEA2 answers with nearest-neighbour density. Those are useful local density estimates, but they are not the quality measure I usually trust for the final set. If I judge an approximation by the dominated hypervolume it covers, then the survival pressure should not be a separate geometric proxy unless I have no alternative.

The direct target would be: after I add new material to the population, keep the fixed-size subset with the largest S-metric value. That sentence is clean, but it hides the cost. If I make a whole offspring population and then choose `mu` points out of `mu + lambda`, the exact subset problem explodes. The first correction is to change the rhythm of the algorithm, not the metric. I should not create a large batch and then solve a large subset problem. I create one offspring, form a pool of size `mu + 1`, and delete exactly one solution. Then the set-level objective becomes a one-delete question.

For a candidate `s` inside a set `R`, the volume I lose by deleting it is
`Delta S(s, R) = S(R) - S(R \ {s})`. This has the sign I want: large positive means `s` owns a region of dominated space that no other member covers, zero means `s` is redundant, and the point to discard is the one with minimum `Delta S`. I have to keep the subtraction in this order. Reversing it would make every useful point look negative and would select the most valuable point for deletion.

I still cannot apply that comparison to the whole pool blindly, because dominance must remain the primary rule. If a point is on a worse nondominated-sorting level, it should be eligible for deletion before any better-front point. So I partition the `mu + 1` pool into fronts `R_1, ..., R_v` and only look at `R_v`, the worst-ranked front. If `R_v` contains one point, the decision is forced. If it contains several mutually nondominated points, I compare their hypervolume contributions within that front and remove the smallest contributor. This keeps the dominance rule outside the indicator rule: rank first, volume contribution second.

Now I need the low-dimensional contribution cases to be exact. In two objectives, a nondominated front sorted by increasing `f_1` is automatically sorted by decreasing `f_2`. For an interior point `s_i`, its unique rectangle begins at its own coordinates, ends in `f_1` at the next point, and ends in `f_2` at the previous point. The contribution is therefore
`(f_1(s_{i+1}) - f_1(s_i)) * (f_2(s_{i-1}) - f_2(s_i))`. Both factors are nonnegative under that ordering. If either factor is small, the point is crowded against a neighbour and is a natural deletion candidate. The two extremes are different: their rectangles reach to the reference point. In the two-objective case I can avoid making the reference point decide the endpoints by protecting the extremes and computing finite contributions only for interior points. If a degenerate front has no interior point, then I fall back to the ordinary finite-reference hypervolume contribution so that the code still returns one deletion.

For three objectives I cannot protect every boundary point. Many points can carry one worst coordinate, and keeping all of them would turn "boundary" into a population-size trap. I need a concrete reference point. The neutral dynamic choice is to recompute it from the current front or pool as the coordinatewise worst objective vector plus `1.0`: `r = nadir + 1.0`. The offset gives boundary boxes positive thickness without adding a scale-dependent span multiplier. That also matches the implementation pattern I want: whenever I need exact contributions beyond the simple 2D interior formula, compute `S(R, r) - S(R \ {s}, r)` with `r` set just beyond the current nadir.

There is one useful variant for dominated worst fronts. If `R_v` is not the first front, all of its points are already dominated by something better. Hypervolume contribution within that bad front can still preserve a point sitting under a sparse region, but it may spend computation on dominated material. A cheaper variant deletes, from the worst front, the point dominated by the largest number of pool members; when the whole pool is nondominated, that count is zero for everyone and the algorithm returns to hypervolume contribution. I should keep this as an option, not confuse it with the basic rule.

The algorithm is now small. Initialize a fixed population. Repeat: generate one child, evaluate it, append it, nondominated-sort the `mu + 1` pool, and delete one member from the worst front. The deletion is the minimum hypervolume contributor for the basic version, or the highest domination count in the dominated-front variant. Since each step deletes only one member, the population size stays fixed and the S-metric of the retained relevant front cannot decrease by accidentally throwing away a better contributor.

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

The important thing is that the code refuses the batch-selection shortcut. `reduce` expects `mu + 1`, not `2 * mu`, because the whole design is a steady-state answer to the cost of hypervolume. The signs are all minimization signs: lower objectives dominate, the reference point is worse, and a contribution is "with minus without." The 2D interior formula uses the next point in `f_1` and the previous point in `f_2`; swapping either neighbour would turn a positive exclusive rectangle into the wrong area. For three objectives the exact contribution falls back to repeated dominated-volume calculations against the dynamic `nadir + 1.0` reference, which is slower but faithful for the dimensions this algorithm is meant to handle.

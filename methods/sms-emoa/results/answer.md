# SMS-EMOA: S-metric selection EMOA

SMS-EMOA is a steady-state elitist multi-objective evolutionary algorithm. It keeps a
population of size `mu`, creates one evaluated offspring, forms a `mu + 1` pool, and
removes exactly one individual. The removal rule combines nondominated sorting with
dominated hypervolume:

1. Sort the `mu + 1` pool into fronts `R_1, ..., R_v`.
2. Only the worst front `R_v` is eligible for deletion.
3. In the basic rule, delete the point with smallest exclusive hypervolume contribution
   `Delta S(s, R_v) = S(R_v) - S(R_v \ {s})`.
4. In the domination-count variant, if there is more than one front, delete from `R_v`
   the point dominated by the largest number of pool members; if the whole pool is
   nondominated, use the hypervolume contribution rule.

For two objectives, sort the worst front by increasing `f_1`; then `f_2` decreases along
the front. Interior contributions are

```text
Delta S(s_i, R_v) = (f_1(s_{i+1}) - f_1(s_i)) * (f_2(s_{i-1}) - f_2(s_i)).
```

The signs are positive for minimization. The two extremes are protected in the
two-objective case; if a degenerate front has no interior point, finite-reference
hypervolume breaks the tie. For three objectives, compute exact contributions with a
dynamic reference point `r = nadir(R_v) + 1.0`.

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

The method's only essential new survival decision is the deletion rule. Variation can remain
ordinary SBX/polynomial mutation or any other evaluated-offspring generator, because the
S-metric selection step operates entirely on the objective vectors in the `mu + 1` pool.

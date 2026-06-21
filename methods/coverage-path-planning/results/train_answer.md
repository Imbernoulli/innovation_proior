The problem is to plan a path for a robot whose working footprint must pass over every point of a bounded free region, not simply travel from a start to a goal. Floor scrubbers, de-mining robots, crop plows, and survey vehicles all share this requirement: the area swept by the footprint along the path must equal the entire free space. A missed strip is unacceptable in safety-critical settings, so the planner must be complete in the motion-planning sense: if a covering path of a connected free component exists, the method must return one in finite time.

Existing ideas fall short in different ways. Hand-programmed paths or landmark-based coverage lack automation and offer no completeness guarantee. Grid-based methods approximate the region at finite resolution, so completeness is only resolution-complete and coverage quality depends on grid fineness. The trapezoidal decomposition is exact and complete for polygonal environments, but it cuts a new cell at every polygon vertex, including vertices where the free-space connectivity does not actually change. Those extra cells force redundant lengthwise passes and turns, which are the dominant cost of executing a coverage path because every turn requires deceleration, rotation, acceleration, and accumulates dead-reckoning error. What is needed is an exact decomposition that creates cells only where the geometry truly forces a split.

The method is Boustrophedon Cellular Decomposition. It is named after the back-and-forth plowing motion, boustrophedon, which covers a simple convex region by parallel runs spaced by the robot footprint. The decomposition extends that motion to general non-convex environments by cutting free space into cells only at critical points where the connectivity of a sweeping slice changes.

The core idea is to sweep a vertical slice left to right across the free space and monitor how many disjoint free intervals the slice intersects at each column. When the interval count increases, the slice has reached the left tip of an obstacle and the free space has split above and below it; this is an IN event, and the current cell must close while two new cells open. When the interval count decreases, the slice has passed the right tip of an obstacle and the space has rejoined; this is an OUT event, and the two cells above and below close while a single new cell opens. When the interval count stays the same, only the top or bottom boundary of the current cell has kinked; this is a MIDDLE event, and rather than start a new cell, the existing cell boundary is updated to follow the kink. Because the cut is keyed to connectivity change rather than to polygon vertices, the method applies not only to polygonal environments but also to curved or sampled free spaces where only free and occupied samples are known.

Each cell produced by this sweep is monotone in the slice direction: every vertical slice through the cell intersects it in a single interval. That is exactly the shape a boustrophedon sweep covers cleanly. Each cell becomes a node in an adjacency graph, with edges between cells that share a boundary created at an IN or OUT event. Coverage then reduces to finding an exhaustive walk over this graph; visiting every node and sweeping its cell guarantees that the union of swept cells, which equals the union of cells, which equals the free space, is fully covered. Finding the cheapest such walk is NP-complete, so the method deliberately settles for completeness rather than optimality. A depth-first traversal with backtracking through already covered cells suffices, and the sweep direction can be varied and the results scored to reduce turns or path length heuristically.

The implementation below works on a raster grid where 1 denotes free space and 0 denotes obstacle. It computes slice connectivity, performs the single left-to-right sweep with IN/OUT event detection, builds the adjacency graph, orders the cells by depth-first search, and then covers each cell with a boustrophedon motion whose parallel runs are spaced by the side-step. Transit between cells is left to a standard point-to-point planner; the decomposition provides the geometry, while sensor feedback is assumed to keep the robot localized during execution.

```python
import numpy as np


def slice_connectivity(slice_column):
    """Maximal free runs of one slice (column). Returns (count, [(start, end), ...]).
    The CHANGE of this count between consecutive slices defines a critical point."""
    connectivity, parts = 0, []
    last, start = 0, None
    for i, v in enumerate(slice_column):
        if last == 0 and v == 1:
            start = i
        elif last == 1 and v == 0:
            connectivity += 1
            parts.append((start, i))
        last = v
    if last == 1:
        connectivity += 1
        parts.append((start, len(slice_column)))
    return connectivity, parts


def slices_overlap(left_parts, right_parts):
    """Adjacency matrix: A[l, r] = 1 where a left interval and a right interval
    share vertical extent (i.e. are connected across the one-column gap)."""
    A = np.zeros((len(left_parts), len(right_parts)), dtype=int)
    for l, (l0, l1) in enumerate(left_parts):
        for r, (r0, r1) in enumerate(right_parts):
            if min(l1, r1) - max(l0, r0) > 0:
                A[l, r] = 1
    return A


def decompose(grid_map):
    """Single left-to-right slice sweep. Open/close cells only where the slice's
    connectivity changes (IN/OUT critical points); MIDDLE events just extend the
    live cell. Returns (seg_map labelled per free pixel, cells, adjacency)."""
    H, W = grid_map.shape
    seg_map = np.zeros_like(grid_map)
    adjacency = {}
    cells = []
    next_id = 1
    last_conn, last_parts, last_ids = 0, [], []

    for x in range(W):
        conn, parts = slice_connectivity(grid_map[:, x])

        if last_conn == 0 and conn > 0:
            ids = []
            for _ in parts:
                cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                adjacency.setdefault(next_id, set())
                ids.append(next_id)
                next_id += 1
        elif conn == 0:
            last_conn, last_parts, last_ids = conn, parts, []
            continue
        else:
            A = slices_overlap(last_parts, parts)
            ids = [0] * len(parts)
            for l in range(A.shape[0]):
                s = A[l].sum()
                if s == 1:
                    ids[int(np.argmax(A[l]))] = last_ids[l]
                elif s > 1:
                    for r in np.flatnonzero(A[l]):
                        cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                        adjacency.setdefault(next_id, set())
                        adjacency[last_ids[l]].add(next_id)
                        adjacency[next_id].add(last_ids[l])
                        ids[r] = next_id
                        next_id += 1
            for r in range(A.shape[1]):
                s = A[:, r].sum()
                if s > 1:
                    cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                    adjacency.setdefault(next_id, set())
                    for l in np.flatnonzero(A[:, r]):
                        adjacency[last_ids[l]].add(next_id)
                        adjacency[next_id].add(last_ids[l])
                    ids[r] = next_id
                    next_id += 1
                elif s == 0:
                    cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                    adjacency.setdefault(next_id, set())
                    ids[r] = next_id
                    next_id += 1

        for cid, (a, b) in zip(ids, parts):
            seg_map[a:b, x] = cid
            cells[cid - 1]["end"] = x
            cells[cid - 1]["bounds"].append((a, b))

        last_conn, last_parts, last_ids = conn, parts, ids

    return seg_map, cells, adjacency


def order_cells(adjacency, start):
    """Depth-first exhaustive walk with backtracking through covered cells."""
    walk, seen = [], set()

    def dfs(node):
        seen.add(node)
        walk.append(node)
        for nb in sorted(adjacency.get(node, ())):
            if nb not in seen:
                dfs(nb)
                walk.append(node)

    dfs(start)
    return walk


def sweep_cell(cell, side_step):
    """Boustrophedon: step the cell's columns by side_step, running the footprint
    up or down and flipping vertical direction each column (the back-and-forth)."""
    path = []
    cols = list(range(cell["start"], cell["end"] + 1))
    going_down = True
    for x, (top, bottom) in zip(cols[::side_step], cell["bounds"][::side_step]):
        ys = list(range(top, bottom, side_step))
        path += [(x, y) for y in (ys if going_down else reversed(ys))]
        going_down = not going_down
    return path


def plan_coverage(grid_map, side_step=1):
    """Return coverage sweeps plus the exhaustive cell walk used for transits."""
    seg_map, cells, adjacency = decompose(grid_map)
    by_id = {c["id"]: c for c in cells}
    walk = order_cells(adjacency, start=cells[0]["id"])
    covered, path = set(), []
    for cid in walk:
        if cid in covered:
            continue
        path += sweep_cell(by_id[cid], side_step)
        covered.add(cid)
    return path, seg_map, walk
```

Two practical caveats remain. Because the side-step is discrete, slivers of free space tucked into acute angles between obstacle boundaries and the sweep direction can fall between runs and be missed; shrinking the side-step trades coverage precision for more passes, and applications such as floor cleaning already handle boundary slivers with a separate wall-following pass. Second, dead-reckoning error accumulates over long coverage paths, so the executor must use sensor feedback to stay registered to the environment. The planner itself supplies the geometric guarantee that, if executed faithfully, the path covers the entire free space.

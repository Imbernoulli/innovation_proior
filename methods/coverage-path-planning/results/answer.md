# Boustrophedon Cellular Decomposition: provably complete coverage path planning

## Problem

Unlike start–goal planning, *coverage* asks for a path whose swept footprint equals an entire bounded free region: a floor scrubber, de-mining robot, or crop plow must pass its working footprint over **every** point, not reach a target. The guarantee demanded is *completeness* in the motion-planning sense — if a covering path of a connected free component exists, the planner returns one in finite time (a missed strip in de-mining is an undetected mine). The region is general non-convex free space with obstacles, and turning is the expensive part of execution (deceleration, dead-reckoning drift), so the path should also be economical in the number of lengthwise passes.

## Key idea

A convex obstacle-free region is covered trivially by **boustrophedon** ("way of the ox") motion: run lengthwise across it, turn, run back along a parallel line shifted by the footprint (the *side-step*), repeat. Side-step = footprint makes the swept strips abut with no gap. Obstacles break the single sweep — a lengthwise run hits an obstacle and the free space splits above/below it — so cut the free space into cells, each boustrophedon-coverable, whose union is **exactly** the free space (an *exact* cellular decomposition, not a grid approximation that is only resolution-complete). With each cell a node and shared boundaries as edges, coverage reduces to *visiting every node of the cell-adjacency graph*; an exhaustive walk always exists, so completeness is free.

The decomposition rule is the contribution. Sweep a vertical line — a *slice* — left to right. The trapezoidal decomposition cuts at every polygon vertex, including **MIDDLE** vertices where the free space neither splits nor merges, manufacturing redundant cells and the extra passes and turns they cost. Instead, cut only at **critical points** — where the slice's *connectivity* (number of free intervals) changes:

- **IN event:** connectivity increases, one interval becomes two (slice reached an obstacle's left tip; free space splits). Close the current cell; open two new cells.
- **OUT event:** connectivity decreases, two intervals become one (obstacle's right tip; free space rejoins). Close the two cells; open one.
- **MIDDLE event:** connectivity unchanged — no cut, just update the current cell's boundary.

Between critical events, each continuing slice interval remains one cell. A one-interval region stays one cell; after an IN event, the upper and lower intervals are two separate live cells until the OUT event rejoins them. MIDDLE vertices only update those cells' wiggly top/bottom boundaries, so each cell stays monotone in the slice direction — exactly what a boustrophedon sweep covers cleanly. Keying on connectivity rather than vertices also lifts the method off polygons onto curved and sampled spaces for free, since "did the interval count change?" is askable anywhere.

## Algorithm

1. **Slice connectivity.** For each column, find the maximal runs of free cells and their `(start, end)` intervals; the count is the connectivity.
2. **Decompose (single left-to-right sweep).** Compare each column's intervals to the previous via vertical overlap. A left interval overlapping exactly one right interval → cell continues, unless the right interval is part of a merge; overlapping several → IN event, open a new cell per right interval; a right interval overlapping several lefts → OUT event, open one new cell; a right interval overlapping none → fresh free space, open a cell. Whenever a cell closes and cells open at the same event, add adjacency edges between them. One pass yields cells + adjacency graph.
3. **Order cells.** Depth-first exhaustive walk over the adjacency graph, including backtracking entries through already covered cells. This gives **completeness, deliberately not optimality** (optimal coverage is NP-complete; cheaper paths can be sought by re-running over a few sweep directions and scoring path length/turns).
4. **Sweep each cell.** Boustrophedon: step through the cell's columns by the side-step, running the footprint up or down and flipping direction each column. Transit through already-covered cells without re-sweeping.

Union of swept cells = union of cells = free space ⟹ provably complete coverage.

## Code

Self-contained boustrophedon cellular decomposition on a raster (`1` = free, `0` = obstacle), with the connectivity-based event logic, single-sweep adjacency-graph construction, depth-first cell ordering, and per-cell boustrophedon sweep.

```python
import numpy as np


def slice_connectivity(slice_column):
    """Maximal free runs of one slice (column). Returns (count, [(start, end), ...]).
    The CHANGE of this count between consecutive slices defines a critical point."""
    connectivity, parts = 0, []
    last, start = 0, None
    for i, v in enumerate(slice_column):
        if last == 0 and v == 1:        # entering a free run
            start = i
        elif last == 1 and v == 0:      # leaving a free run -> one interval closed
            connectivity += 1
            parts.append((start, i))
        last = v
    if last == 1:                       # run reaches the bottom edge
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
    seg_map = np.zeros_like(grid_map)        # cell id per free pixel (0 = obstacle)
    adjacency = {}                           # cell id -> set of adjacent cell ids
    cells = []                               # cell id -> {id, start, end, bounds}
    next_id = 1
    last_conn, last_parts, last_ids = 0, [], []

    for x in range(W):
        conn, parts = slice_connectivity(grid_map[:, x])

        if last_conn == 0 and conn > 0:
            # slice just entered free space: open one cell per interval
            ids = []
            for _ in parts:
                cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                adjacency.setdefault(next_id, set())
                ids.append(next_id); next_id += 1
        elif conn == 0:
            # slice left the free space: everything open just closed
            last_conn, last_parts, last_ids = conn, parts, []
            continue
        else:
            A = slices_overlap(last_parts, parts)
            ids = [0] * len(parts)
            for l in range(A.shape[0]):
                s = A[l].sum()
                if s == 1:                       # cell continues
                    ids[int(np.argmax(A[l]))] = last_ids[l]
                elif s > 1:                       # IN event: free space split
                    for r in np.flatnonzero(A[l]):
                        cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                        adjacency.setdefault(next_id, set())
                        adjacency[last_ids[l]].add(next_id)   # closing cell shares a boundary
                        adjacency[next_id].add(last_ids[l])
                        ids[r] = next_id; next_id += 1
            for r in range(A.shape[1]):
                s = A[:, r].sum()
                if s > 1:                         # OUT event: cells merged
                    cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                    adjacency.setdefault(next_id, set())
                    for l in np.flatnonzero(A[:, r]):
                        adjacency[last_ids[l]].add(next_id)
                        adjacency[next_id].add(last_ids[l])
                    ids[r] = next_id; next_id += 1
                elif s == 0:                      # fresh free space: open a cell
                    cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                    adjacency.setdefault(next_id, set())
                    ids[r] = next_id; next_id += 1

        # label this column and grow each live cell's extent + boundary
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
        going_down = not going_down          # the turn at the end of each pass
    return path


def plan_coverage(grid_map, side_step=1):
    """Return coverage sweeps plus the exhaustive cell walk used for transits."""
    seg_map, cells, adjacency = decompose(grid_map)
    by_id = {c["id"]: c for c in cells}
    walk = order_cells(adjacency, start=cells[0]["id"])
    covered, path = set(), []
    for cid in walk:
        if cid in covered:
            continue                          # the walk records transit; don't re-cover
        path += sweep_cell(by_id[cid], side_step)
        covered.add(cid)
    return path, seg_map, walk
```

The discrete side-step can skip slivers where an obstacle boundary meets the runs at an acute angle (shrink the side-step, or add a dedicated boundary pass); and over a long path dead-reckoning error accumulates, so the executor needs sensor feedback to stay registered — the planner supplies the geometry, not the localization.

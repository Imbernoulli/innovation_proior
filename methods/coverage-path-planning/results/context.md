# Context: Planning a path that passes a robot's footprint over every point of a region

## Research question

Classical motion planning answers the *start–goal* question: given a start and a goal configuration, find a collision-free path between them. A large and growing family of robot tasks asks something different. A floor scrubber, a robotic vacuum, a de-mining robot, a crop plow, an oceanographic survey vehicle, a car-body painter, a milling tool — none of these has a goal point. Each must drag its working footprint (a brush, a cutter, a mine detector, a camera, a spray) over **every point** of a bounded region, so that the area swept out equals the target region. This is *coverage path planning*: integrate the robot's footprint along the path and the result must be the whole free space.

The pain point is sharpest in safety-critical coverage. In de-mining, a gap in the swept region is an undetected mine; "we probably covered most of it" is not acceptable. So the planner must be **complete** in the motion-planning sense (Latombe 1991): in finite time it must return a path that covers a connected component of the robot's free space whenever one exists. Most coverage planners in use are rudimentary — they follow heuristics or replay paths programmed by hand or by deployed landmarks — and offer no such guarantee. The task is to produce, automatically and provably, a path that covers a general non-convex region populated by obstacles, while keeping the path cheap to execute.

Coverage is related to, but harder than, the *covering salesman problem* — a traveling-salesman variant where the agent need only visit a *neighborhood* of each city. In coverage the agent must pass over *all* points of the region, not merely through neighborhoods of a finite set, so the combinatorial structure that makes TSP tractable to approximate does not directly transfer.

## Background

**Start–goal and map-based planning.** Motion planning matured on the start–goal problem, with potential-function methods, sampling, and provably complete roadmap methods. Yap and Canny introduced map-based planning: build a map once (a fixed cost), then answer many start–goal queries cheaply against it. None of this machinery addresses coverage — these planners optimize a path *through* the space, not the area *swept* by the robot.

**The footprint and the side-step.** A coverage path is built from straight runs spaced by the robot's footprint width. If consecutive parallel runs are spaced by the footprint, their swept strips tile the region with no gaps; if spaced wider, strips of floor are missed. The spacing (the *side-step*) is therefore the knob that trades completeness against the number of runs.

**The back-and-forth (boustrophedon) motion.** The natural way to cover a simple region is the lawnmower pattern: traverse the full length of the region in a straight line, turn around, traverse an adjacent parallel line in the opposite direction, and repeat. This is the *boustrophedon* motion — Greek for "the way of the ox," after an ox dragging a plow across a field, a term in English use since 1699. For a convex, obstacle-free region this single sweep is trivially complete. The difficulty is everything that is *not* a simple convex region.

**Cellular decomposition.** The classical tool for non-convex free space is to break it into pieces. An *exact cellular decomposition* is a set of non-intersecting regions (cells) whose union is exactly the free configuration space. Represent each cell as a node of an *adjacency graph*, with an edge between cells that share a boundary. If every individual cell is simple enough to be covered by a known motion, then covering the whole region reduces to a graph problem: find a walk through the adjacency graph that visits every node at least once. Such a walk is a traveling-salesman-flavored problem over the cells, for which a (possibly sub-optimal) solution always exists.

**The trapezoidal (slab) decomposition.** The popular exact decomposition for a polygonal environment is the trapezoidal, or slab, decomposition (Latombe 1991; Preparata & Shamos 1985). A vertical line — a *slice* — sweeps from left to right across the bounded environment. The slice intersects the free space in one or more intervals. As the slice passes a vertex of an obstacle polygon, an *event* occurs, classified as **IN**, **OUT**, or **MIDDLE**: loosely, at an IN event the current cell closes and two new cells open (the free space splits around an obstacle); at an OUT event two cells close and one opens (the free space rejoins past an obstacle); a MIDDLE event is an intermediate vertex. The result is that the free space is carved into trapezoidal cells, each of which — being a trapezoid — is covered by simple back-and-forth motions, and coverage is achieved by visiting every cell of the adjacency graph. Its gap: because *every* polygon vertex produces an event and thus a cell boundary, the decomposition has many cells, and many cells mean many redundant lengthwise passes. Two adjacent trapezoids that are each, say, two-and-a-half robot-widths wide force `ceil(2.5) + ceil(2.5) = 6` lengthwise motions — the side-step is rounded up once per cell, so an extra cell boundary pays a rounding penalty even when the geometry across it is uniform. Fewer cells is better, because turning is the expensive part of a coverage path: the robot must decelerate, turn, accelerate, and each turn accrues dead-reckoning error. The trapezoidal approach also assumes the environment is polygonal.

**Critical points and roadmaps.** Roadmap motion planning (Canny 1988; Canny & Lin's Opportunistic Path Planner, 1990, 1993, itself built on Canny's roadmap algorithm) introduced *critical points* — distinguished configurations of a swept slice — as a way to structure free space, and this machinery applies to continuous spaces including curved and even sampled environments rather than only polygonal ones.

**Prior coverage work and its limits.** Early floor-maintenance work (Colegrave & Branch 1994) required the coverage path to be programmed into the robot by hand or to rely on landmarks deployed in the environment — no automatic path generation. The Demeter agricultural project (Ollis & Stentz 1996) used vision to follow the previous cut crop-line and could only cover rectangular fields. A template-based floor coverage that respects non-holonomic constraints (Hofner & Schmidt 1995) covers a single obstacle-free bounded region but gives no method to compose templates into full coverage when obstacles are present. Zelinsky et al. (1993) gave a complete coverage method for unstructured environments, but on a *discretized* (resolution-complete) grid. VanderHeide & Rao (1995) built a sensor-based (on-line) trapezoidal decomposition for an environment with one or two well-separated obstacles, inheriting the trapezoidal redundancy and the requirement of few, separated obstacles. Hert et al. (1996) produced a terrain-covering algorithm for an AUV that, in the planar case, traces nearly the same back-and-forth paths, but it is incremental and not proven complete. Kurabayashi et al. (1996) sketched a cooperating-robot version without proof. Across these, no single method is simultaneously *complete*, *automatic*, applicable to *general* non-convex (and ideally non-polygonal) environments, and economical in the number of lengthwise passes.

**Diagnostic facts about real execution.** Two facts about the world and the robot bound any practical coverage scheme. First, the side-step is discrete, so regions near an obstacle boundary that forms an acute angle with the back-and-forth runs can be skipped — a shorter side-step reduces the missed area at the cost of more passes (a time/coverage tradeoff). Second, a mobile base accumulates dead-reckoning error over a long coverage path; without sensor feedback the estimated pose drifts far enough to drive the robot into an obstacle. Floor-cleaning practice already treats the region near walls separately, with a dedicated boundary pass.

## Baselines

**Trapezoidal / slab decomposition (Latombe 1991; Preparata & Shamos 1985).** Sweep a vertical slice left to right across a polygonal environment; open/close cells at IN/OUT/MIDDLE events triggered by polygon vertices; each resulting trapezoid is covered by back-and-forth motions; visit all cells via the adjacency graph. Gap: one cell boundary per vertex yields many small cells and many redundant lengthwise passes; restricted to polygonal environments.

**Sensor-based trapezoidal coverage (VanderHeide & Rao 1995).** An on-line trapezoidal decomposition that uses line-of-sight sensing, demonstrated for a planar environment with one or two well-separated obstacles. Gap: inherits trapezoidal redundancy and assumes few, separated obstacles.

**Approximate (grid) decomposition (Elfes; Moravec; Zelinsky et al. 1993).** Represent free space by a fine uniform grid; assume a cell is covered once entered (cell ≈ footprint); cover by visiting every cell. Zelinsky et al. use a wavefront/distance-transform potential filled over the whole free space and descend a "pseudo-gradient" to a coverage path, with extra potential terms (e.g. obstacle distance) to bias toward fewer turns. Gap: the decomposition only *approximates* the region (its union is not exactly the free space), so completeness is resolution-complete, not exact, and coverage quality is tied to grid resolution.

**Spanning-Tree Coverage (Gabriely & Rimon).** Subdivide the work-area into disjoint cells and follow a spanning tree of the cell-induced graph, covering every point exactly once; off-line, on-line, and ant-like variants run in O(N) (some in O(1) memory). Gap: requires the free space never be narrower than roughly twice the robot/tool diameter, and is grid-based.

**Semi-approximate coverage (Hert & Lumelsky).** A partial discretization: cells are fixed in width but the ceiling and floor of each may be any shape. The robot zigzags along parallel grid lines and recursively covers any *inlets* it would otherwise miss or double-cover, remembering inlet doorways so each is covered once; islands are handled by converting their surroundings into artificial inlets. Proven correct, on-line, with path length linear in boundary and grid-line lengths. Gap: cells fixed in width is an intermediate structure between approximate and exact; it is not a fully exact decomposition keyed to the geometry of the obstacles.

**Hand-programmed / landmark / template coverage (Colegrave & Branch 1994; Hofner & Schmidt 1995).** Paths programmed by hand or via deployed landmarks, or templates for a single obstacle-free region. Gap: not automatic and/or no composition rule for obstacles; no completeness guarantee.

## Evaluation settings

The natural yardstick is a bounded, connected planar free space — a floor plan bounded above and below by line segments and populated with obstacles — both in simulation and on a real mobile base (e.g. a Nomadic Technologies Nomad 200, with obstacles laid out as polygons in the robot's display). Representative environments span a small room (on the order of a few hundred square feet) and a larger one (on the order of several hundred to a thousand square feet) with two or more obstacles. The robot is modeled with a fixed footprint and a chosen side-step; the slice/sweep direction is a free parameter. The criteria are: whether the full free space is covered (the completeness property), the number of lengthwise back-and-forth passes (a proxy for execution cost, since turns dominate time and induce dead-reckoning error), and, on the real base, how the commanded path diverges from the executed path as dead-reckoning error accumulates.

## Code framework

The primitives that already exist: a raster/occupancy representation of the bounded free space (free vs. obstacle), the notion of a vertical *slice* (one column) and of the connected free intervals within a slice, an adjacency-graph container with a generic graph-search routine, and a per-cell back-and-forth (boustrophedon) sweep that steps by the footprint side-step. The missing rules are how the sequence of slices becomes cells, and how the per-cell sweeps are ordered into one coverage path.

```python
import numpy as np

# free space as a raster: 1 = free, 0 = obstacle
# a "slice" is one column grid_map[:, col]

def slice_connectivity(slice_column):
    """Connected free intervals of one slice.
    Returns (count, [(start, end), ...])."""
    pass  # TODO: scan the column for maximal runs of free cells

def slices_overlap(left_intervals, right_intervals):
    """Which intervals of two consecutive slices vertically overlap
    (i.e. are connected across the one-column gap)."""
    pass  # TODO: adjacency between segments of neighbouring slices

def decompose(grid_map):
    """Sweep the slice left to right; return a labelled cell map,
    cell records, and the cell adjacency graph."""
    pass  # TODO: open/close cells as the slice sweeps; build adjacency graph

def order_cells(adjacency_graph, start_cell):
    """A walk over the adjacency graph that visits every cell,
    including backtracking entries through already visited cells."""
    pass  # TODO: graph search producing an exhaustive cell walk

def sweep_cell(cell, side_step):
    """Cover one cell with back-and-forth passes spaced by side_step."""
    pass  # TODO: boustrophedon motion within a single cell

def plan_coverage(grid_map, side_step):
    cell_labels, cells, graph = decompose(grid_map)
    walk = order_cells(graph, start_cell=cells[0]["id"])
    by_id = {cell["id"]: cell for cell in cells}
    covered = set()
    path = []
    for cell_id in walk:
        if cell_id in covered:
            continue
        path += sweep_cell(by_id[cell_id], side_step)
        covered.add(cell_id)
    return path, cell_labels, walk
```

OK, let me think this through from scratch. I have a robot — a floor scrubber, a de-mining vehicle, a crop plow, doesn't matter — and the job is not "get from A to B." The job is: every point of this region has to be passed over by the robot's working footprint. The brush, the mine detector, the plow blade. When I'm done, the area swept out by the footprint along my path has to equal the whole free region, with nothing missed.

That's a strange goal compared to everything in the planning literature I know. Start–goal planning — potential fields, roadmaps, Canny's provably complete methods — all of it optimizes a path *through* the space. The path is a curve; the cost is its length or its clearance. Here I don't care about a curve reaching a target. I care about the *area* my footprint integrates over. There's no goal point at all. So I can't just import a start–goal planner and call it a day; the objective is a different animal.

Let me name what "done" has to mean, precisely, because in some of these applications sloppiness is fatal. De-mining: a strip of floor I failed to sweep is a mine I didn't find. So "I probably got most of it" is worthless. I need *completeness* in the strong sense — if a covering path of this connected free region exists, my method finds one in finite time, guaranteed, not just usually. Most of the coverage I've seen in practice is heuristic — follow-the-wall rules, replay a hand-programmed path, drop landmarks and chase them. Those give no guarantee. Using a heuristic for de-mining is like using a mine detector that works most of the time. Not good enough. The bar is a provable guarantee.

Is this just a salesman problem in disguise? There's the covering-salesman variant — visit a *neighborhood* of each city, minimize travel. Tempting to map "cities" to points of the region. But the mismatch is exactly the thing that makes coverage hard: I must pass *over* all points, a continuum, not merely *through* the neighborhoods of a finite set. A continuum of constraints. I can't enumerate points and route between them.

So let me start where it's easy and see how far it gets me. Take the simplest possible region: a convex blob, no obstacles. How do I cover that? The dumb, obviously-correct thing: pick a direction, run all the way across the region in a straight line, turn around, run back along a parallel line shifted over by one footprint-width, turn, run again. The lawnmower pattern. The ox dragging a plow does exactly this — crosses the field, turns at the end, comes back one furrow over. "Boustrophedon," the way of the ox. For a convex obstacle-free region this is trivially complete: as long as each run is spaced from the last by the footprint width, the swept strips abut with no gap, and the region being convex means every run goes clean from one boundary to the other. Done. No cleverness needed.

The spacing matters and I should pin it down now, because it's the one real knob in this simple picture. Call it the side-step — the perpendicular shift between consecutive runs. If the side-step equals the footprint width, consecutive swept strips just touch, tiling the region with no overlap and no gap. Make the side-step bigger and I leave unswept ribbons between strips — incomplete. Make it smaller and I cover everything but with overlap and more passes. So side-step = footprint is the natural choice for completeness; I'll hold onto the fact that shrinking it is a lever I can pull later if I find I'm missing slivers somewhere.

Now break the easy case. The region is non-convex and there are obstacles in it — pillars, furniture, whatever. Try the same single boustrophedon sweep across the whole thing and watch it fail: a straight run that's supposed to go from one side to the other slams into an obstacle partway. I can't complete the lengthwise pass. And around the obstacle the free space has split — above it and below it — and a single back-and-forth pattern doesn't know which part it's in. So the one-sweep idea breaks the moment the free space stops being a simple shape.

The fix that suggests itself: if I can't sweep the whole thing as one simple region, cut the free space into pieces, each of which *is* a simple region I can sweep. Cover every piece, and I've covered the whole. This is the classical move for non-convex free space — a cellular decomposition. And I want an *exact* one: a set of non-overlapping cells whose union is *exactly* the free space, not an approximation of it. That word "exact" is load-bearing. If I instead lay a fine grid over the region and call each grid square a cell, the union of cells only *approximates* the region — completeness becomes resolution-complete, true only down to the grid spacing, and tied to how fine I'm willing to make the grid. For a de-mining guarantee I don't want "complete down to 5cm." I want the cells to *be* the free space.

What do I get from an exact decomposition? Make each cell a node in a graph; put an edge between two cells that share a boundary — the adjacency graph. Now here's the leverage: if every cell is shaped so that it's coverable by a simple boustrophedon sweep, then the whole coverage problem collapses to a graph problem. Cover the region ⟺ visit every node of the adjacency graph and sweep its cell. "Visit every node at least once" — that's a walk over the graph, a traveling-salesman flavor over the cells. And a walk visiting all nodes always exists (a sub-optimal one is trivial — wander the graph until you've hit them all). So if I can build the right decomposition, completeness is *free*: it's just "did I visit every cell?", and that's decidable and achievable.

So the whole game is now: find a decomposition where (a) the cells exactly tile the free space and (b) every cell is boustrophedon-coverable. Everything rides on the decomposition.

There's a known exact decomposition for polygonal environments — the trapezoidal, or slab, decomposition. Let me work through it because it's the natural thing to reach for and I want to see exactly where it hurts. Take a vertical line — call it a *slice* — and sweep it left to right across the bounded environment. At any position the slice cuts the free space in one or more vertical intervals. As the slice moves and passes a *vertex* of an obstacle polygon, something happens — an event. There are three kinds. At an IN event the slice reaches the left tip of an obstacle: the single free interval there splits into two (one above the obstacle, one below), so the current cell closes and two new cells open. At an OUT event the slice passes the right tip of an obstacle: the two intervals rejoin into one, so two cells close and one opens. A MIDDLE event is any other vertex the slice crosses. Carve a cell boundary at every event and you get a tiling by trapezoids. Each trapezoid is convex-ish and monotone, so each is coverable by simple back-and-forth motions, and visiting every trapezoid in the adjacency graph covers the environment. This works. It's exact, it's complete.

So why am I not done? Stare at how many cells it makes. Every single polygon vertex triggers an event and thus a cell boundary — including the MIDDLE events, the vertices where *nothing structural happened*, where the free space neither split nor merged. Those vertices still slice off a new trapezoid. The decomposition is littered with cells that exist only because some boundary had a kink, not because the connectivity of the space changed.

And cells are not free — each cell costs me back-and-forth passes, and the *boundaries between cells* cost me the worst thing in a coverage path: extra lengthwise motions and extra turns. Let me make that concrete. Picture two trapezoidal cells sitting side by side, sharing a vertical boundary, each about two-and-a-half robot-widths tall. To cover one trapezoid by back-and-forth I need three passes (two-and-a-half rounded up). The neighbor needs three. Six lengthwise motions total. Now suppose those two trapezoids were really one tall region that some irrelevant vertex split in half — if I'd left them merged as a single cell two-and-a-half-ish widths tall, I'd cover it in... still about three passes for the height, but I don't pay the seam: I don't finish the left half, transit across the boundary, and restart the pattern on the right half. Merged, the same area comes out to five passes instead of six. Fewer cells, fewer passes.

Why do I care so much about shaving one pass? Because turning is the expensive part of a coverage path, not the straight runs. Every turn, the robot has to decelerate, pivot, accelerate again — that's where the time goes. Worse, on a real base every turn injects dead-reckoning error; the more turns, the more the robot's estimate of where it is drifts from where it actually is. So redundant cells aren't a cosmetic inefficiency, they directly cost execution time and navigational accuracy. The trapezoidal decomposition is *correct* but spendthrift: it manufactures cells the geometry doesn't actually demand. (And it only works on polygonal environments at all, because it keys on polygon vertices — set that aside for a second, it'll matter.)

So the question sharpens: which cell boundaries do I actually *need*? When is a cut genuinely required? A cut is required exactly when the free space changes its connectivity — when the slice goes from one interval to two (an obstacle starts and the space splits around it) or from two to one (the obstacle ends and the space rejoins). Those are the IN and OUT events. At those moments I *have* to start or stop cells, because a single boustrophedon pass cannot straddle the split — it would have to teleport over the obstacle. But a MIDDLE event? The slice still cuts the free space in the same number of intervals before and after; the connectivity didn't change. A back-and-forth pass can sail right through a MIDDLE event without ever hitting an obstacle. There is no reason to start a new cell there.

So: only open and close cells at the events where the slice's *connectivity changes* — the IN and OUT events — and at a MIDDLE event, don't cut anything; just *update* the current cell's boundary to follow the kink. The consequence is exactly what I wanted: every stretch of the sweep between an IN and the next OUT, where the connectivity holds steady, collapses into a *single* cell. That cell can have a wiggly top and bottom (it absorbed all the MIDDLE events as boundary updates), but it's a monotone region in the slice direction — every vertical slice through it is one unbroken interval — and a monotone region is exactly what a boustrophedon sweep covers cleanly. So the merged cell is still simple to cover, and I've thrown away all the seams the trapezoidal method charged me for. Take the same obstacle that the trapezoidal method chopped into a left cell and a right cell plus the surrounding pieces: now there's one cell to the left of the obstacle, one to the right, and the strips above and below merge straight through. Fewer cells. Fewer lengthwise passes. The decomposition now reflects the *topology* of the free space — where it actually splits and rejoins — and nothing else.

Now the move that pays an unexpected dividend. I keyed the cut on "the connectivity of the slice changes," not on "the slice hits a polygon vertex." Those happened to coincide for a polygonal world, but the connectivity criterion doesn't care about polygons at all. It only asks: did the number of free intervals the slice cuts go up or down? That's a question I can ask of a curved obstacle, a blobby obstacle, even a sampled environment where I only know free/occupied at sample points. The place where the slice's connectivity changes is a *critical point* — and that's not a new idea, it's the same notion that roadmap planning leans on (Canny's roadmap algorithm, and Canny & Lin's Opportunistic Path Planner built on it, structure free space around exactly these connectivity-change points). So by defining events as connectivity changes rather than as vertices, I've quietly generalized the trapezoidal decomposition off of polygons entirely. The cells are defined by critical points, and critical points live in curved and sampled spaces just fine. That's a strict gain over the trapezoidal method I started from, and I got it for free by asking the right question about *when* a cut is necessary.

Let me make the event bookkeeping exact, because I'll have to code it. Sweep the slice from left to right, starting before the free space (from "negative infinity"). The first time the slice touches the free region, the first cell opens. Then, as the slice advances:

— An IN event: the slice's connectivity *increases*, one interval becomes two (the slice reached the left tip of an obstacle and the free space split above and below it). The current cell closes; two new cells open, one for each new interval.

— An OUT event: the slice's connectivity *decreases*, two intervals become one (the slice passed the right tip of an obstacle and the free space rejoined). The two current cells close; one new cell opens.

— A MIDDLE event: connectivity unchanged; just update the current cell — no open, no close.

— Termination: when the slice leaves the bounded free space, close everything still open.

And while I'm sweeping I get the adjacency graph for free, in the same pass. Each cell I open is a node. Whenever a cell closes and new cells open at the same event, the closing cell is adjacent to each opening cell — they share that slice as a boundary — so I draw edges between them. By the time the slice exits, I have both the cells and the full adjacency graph, built incrementally as I went. One left-to-right sweep produces the entire structure.

Now coverage. I have the cells and their adjacency graph; I need an order in which to visit them. The honest thing would be to find the *cheapest* exhaustive walk — minimize total travel and turns over all orderings. But determining the truly optimal coverage path is NP-complete; under even fairly restrictive assumptions it may be intractable. I am not going to solve an NP-complete problem inside the planner just to save a few transit moves. So I'll be pragmatic: I only need *an* exhaustive walk, and any walk that visits every node gives me completeness, which is the property I actually promised. A depth-first-like search over the adjacency graph does it — descend into a cell, recurse into its unvisited neighbors, backtrack when stuck — and it spits out a path list that visits every cell at least once. Completeness comes from "every node visited"; optimality I'm explicitly *not* claiming, and that's the right trade, because the guarantee I need (cover everything) is cheap and the guarantee I'd like (cover everything *optimally*) is NP-complete. If I want to do better than DFS later without paying the NP price, I can vary a parameter — the sweep direction, say — re-run this cheap pipeline, score each resulting path by a metric like total length or turn count, and keep the best. A user can "try" a handful of sweep directions and converge on a near-optimal path with little effort, precisely *because* the underlying decompose-and-walk is so cheap.

Now turn the path list into actual robot motion. Walk the list cell by cell. When I enter a cell that hasn't been covered yet — an "uncleaned" cell — I plan the boustrophedon sweep inside it (back-and-forth, side-step equal to the footprint), then plan a short path from where I finished to the next cell in the list. When I enter a cell that's already been covered — the walk passes back through it as connective tissue — I don't re-sweep; I just transit through it to the next cell. Repeat down the list until it's exhausted. Every cell ends up swept once; every cell swept ⟹ the whole free space covered. That's the completeness argument, end to end: the decomposition's union *is* the free space, the walk hits every cell, and each cell's boustrophedon sweep covers that cell, so the union of swept area is the union of cells is the free space.

Two things nag at me about the real robot, and they're worth being honest about because they shape the implementation. First: the side-step is *discrete*. Where an obstacle boundary cuts across my back-and-forth runs at an acute angle, the corner of free space tucked into that acute wedge can fall between two of my discrete runs and get skipped. It's the same completeness-vs-spacing knob from the convex case, biting near angled boundaries: a shorter side-step shrinks these missed wedges, at the cost of more passes — a time/coverage tradeoff. I won't pretend the discrete sweep is geometrically perfect; I'll note that for applications like floor cleaning the standard remedy is a separate boundary pass — follow each obstacle's edge once after the main coverage — which is consistent with how floor cleaners already treat the region near walls with a different mechanism anyway. Second: over a long coverage path on a real base, dead-reckoning error accumulates — every turn adds drift — and if I plan purely open-loop, the robot's belief about its pose will eventually wander far enough off that it drives into an obstacle. That's not a flaw in the decomposition; it's a fact about odometry, and it says the executor needs sensor feedback (vision, the previous pass's edge à la crop-line following) to stay registered. The planner's job is the geometry; the executor's job is to not get lost while running it.

Let me write it. I'll work on a raster: the free space as a grid, 1 = free, 0 = obstacle. A slice is one column. The connectivity of a slice is the number of maximal runs of free cells in that column, and I want their intervals.

```python
import numpy as np


def slice_connectivity(slice_column):
    # The slice is one grid column. Its connectivity = number of maximal
    # runs of free cells; return that count and each run's (start, end).
    # This is the quantity whose CHANGE defines a critical point / event.
    connectivity = 0
    parts = []
    last = 0
    start = None
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
```

Adjacency between two consecutive slices: two intervals are connected across the one-column gap iff they vertically overlap. That overlap test is what tells me, at each step, whether a free interval continued, split (IN), or merged (OUT).

```python
def slices_overlap(left_parts, right_parts):
    # adjacency matrix: 1 where a left interval and a right interval overlap
    A = np.zeros((len(left_parts), len(right_parts)), dtype=int)
    for l, (l0, l1) in enumerate(left_parts):
        for r, (r0, r1) in enumerate(right_parts):
            if min(l1, r1) - max(l0, r0) > 0:   # they share vertical extent
                A[l, r] = 1
    return A
```

Now the decomposition itself — the left-to-right sweep that opens/closes cells at connectivity changes and labels every free interval of every column with its cell id. The event logic falls straight out of the overlap matrix. If a left interval overlaps exactly one right interval, the cell just continues. If a left interval overlaps *several* right intervals, the free space split there — that's an IN event — so each of those right intervals starts a brand-new cell. If a right interval overlaps *several* left intervals, several cells merged into it — an OUT event — so it starts a new cell too. And a right interval overlapping *no* left interval is free space appearing fresh — open a new cell.

```python
def decompose(grid_map):
    # grid_map[y, x]: 1 = free, 0 = obstacle. Sweep the slice (column) x = 0..W-1.
    H, W = grid_map.shape
    seg_map = np.zeros_like(grid_map)        # cell id labelled per free cell
    adjacency = {}                           # cell id -> set of adjacent cell ids
    cells = []                               # cell id -> {start_col, end_col, bounds}
    next_id = 1                              # 0 is reserved for obstacles
    last_conn, last_parts, last_ids = 0, [], []

    for x in range(W):
        conn, parts = slice_connectivity(grid_map[:, x])

        if last_conn == 0:
            # slice just entered free space: open one cell per interval
            ids = []
            for p in parts:
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
            # left interval -> exactly one right interval: the cell continues
            for l in range(A.shape[0]):
                if A[l].sum() == 1:
                    r = int(np.argmax(A[l]))
                    ids[r] = last_ids[l]
                elif A[l].sum() > 1:                 # IN event: free space split
                    for r in np.flatnonzero(A[l]):
                        cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                        adjacency.setdefault(next_id, set())
                        adjacency[last_ids[l]].add(next_id)   # closing cell ~ each new cell
                        adjacency[next_id].add(last_ids[l])
                        ids[r] = next_id; next_id += 1
            for r in range(A.shape[1]):
                if A[:, r].sum() > 1:                # OUT event: cells merged
                    cells.append({"id": next_id, "start": x, "end": x, "bounds": []})
                    adjacency.setdefault(next_id, set())
                    for l in np.flatnonzero(A[:, r]):                 # both parents ~ new cell
                        adjacency[last_ids[l]].add(next_id)
                        adjacency[next_id].add(last_ids[l])
                    ids[r] = next_id; next_id += 1
                elif A[:, r].sum() == 0:             # fresh free space: open a cell
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
```

The exhaustive walk over the cells — a depth-first traversal. Completeness, not optimality: every cell gets visited.

```python
def order_cells(adjacency, start):
    order, seen = [], set()

    def dfs(node):
        seen.add(node)
        order.append(node)
        for nb in sorted(adjacency.get(node, ())):
            if nb not in seen:
                dfs(nb)
    dfs(start)
    return order
```

Covering one cell with the boustrophedon motion: walk the cell's columns left to right; in each column run the footprint up or down, alternating direction every column so it's genuinely back-and-forth, stepping by the side-step.

```python
def sweep_cell(cell, side_step):
    # back-and-forth over the cell's columns, flipping vertical direction each column
    path = []
    cols = list(range(cell["start"], cell["end"] + 1))
    going_down = True
    for x, (top, bottom) in zip(cols[::side_step], cell["bounds"][::side_step]):
        ys = range(top, bottom, side_step)
        path += [(x, y) for y in (ys if going_down else reversed(list(ys)))]
        going_down = not going_down          # the turn at the end of each pass
    return path
```

And the whole pipeline: decompose, order, then sweep each uncovered cell in turn (transiting through already-covered cells without re-sweeping).

```python
def plan_coverage(grid_map, side_step=1):
    seg_map, cells, adjacency = decompose(grid_map)
    by_id = {c["id"]: c for c in cells}
    order = order_cells(adjacency, start=cells[0]["id"])
    covered, path = set(), []
    for cid in order:
        if cid in covered:
            continue                          # already swept: pass through, don't re-cover
        path += sweep_cell(by_id[cid], side_step)   # plus a transit to the next cell
        covered.add(cid)
    return path, seg_map
```

The causal chain, start to end: the task is to pass a footprint over every point, not to reach a goal, so I anchor on swept *area* and on completeness. A simple convex region falls to a back-and-forth boustrophedon sweep spaced by the footprint. Obstacles break the single sweep, so I cut the free space into cells whose union is *exactly* the free space and cover each cell by a sweep — coverage becomes "visit every node of the cell-adjacency graph." The trapezoidal decomposition does this but cuts at every polygon vertex, manufacturing redundant cells and the extra lengthwise passes and turns they cost; so I cut only where it's *necessary* — where the sweeping slice changes connectivity, the IN and OUT critical points — merging every MIDDLE-event stretch into one monotone, boustrophedon-coverable cell. Keying on connectivity rather than vertices also lifts the method off polygons onto curved and sampled spaces for free. I build the cells and their adjacency graph in a single left-to-right slice sweep, take a depth-first exhaustive walk over the graph (complete, deliberately not optimal, since optimal coverage is NP-complete), and sweep each uncovered cell with footprint-spaced back-and-forth motions, transiting through already-covered cells. Union of swept cells = union of cells = free space: provably complete coverage.

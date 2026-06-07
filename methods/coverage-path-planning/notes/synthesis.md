# Synthesis — Boustrophedon Cellular Decomposition (BCD)

## Sources retrieved & read (this run)
1. PRIMARY: Choset, "Coverage of Known Spaces: The Boustrophedon Cellular Decomposition," Autonomous Robots 9:247-253, 2000 (CMU biorobotics PDF). This is the faithful journal version of Choset & Pignon 1998 FSR "Coverage Path Planning: The Boustrophedon Cellular Decomposition" — same method, same author, same figures/equations. Read in full (7 pp, image scan, read visually).
2. BACKGROUND/BASELINES: Choset, "Coverage for robotics — A survey of recent results," Annals of Math & AI 31:113-126, 2001 (CMU RI PDF). Read in full (14 pp). Gives the 4-category taxonomy and elaborated baselines.
3. EXPLAINER + CODE: AtsushiSakai/PythonRobotics grid-based sweep CPP planner (boustrophedon sweep with turning window). Karlish-git/boustrophedon_cellular_decomposition: real slice/connectivity decomposition (calc_connectivity, get_adjacency_matrix, IN/OUT open/close), DFS adjacency-graph walk, boustrophedon per-cell move. Cloned + read.

## The problem (re-derived in-frame)
- Coverage path planning: find a path so the robot's footprint passes over EVERY point of its free space — not point-to-point (start->goal). Applications: de-mining, floor scrubbing/vacuuming, crop plowing, oceanographic mapping, bridge/inspection, painting.
- "Complete" in the motion-planning sense (Latombe 1991): in finite time finds a coverage path of a connected component of free space, if one exists.
- Related to the covering/watchman salesman problem: visit a neighborhood of each "city"; here must pass OVER all points, not just through neighborhoods.

## Key chain
1. Simple region (convex, obstacle-free): cover with back-and-forth "boustrophedon" (way of the ox, English 1699) lawnmower sweep, passes spaced by footprint width. Trivially complete.
2. General non-convex w/ obstacles: can't sweep in one shot — a single lengthwise pass would hit an obstacle. DECOMPOSE free space into cells, each coverable by a simple boustrophedon sweep.
3. Exact cellular decomposition (Latombe): union of non-intersecting cells = exactly the free space. Each cell -> node in adjacency graph; edge between adjacent cells. If every cell is boustrophedon-coverable, coverage reduces to a walk visiting every node (an exhaustive walk; TSP / Chinese-postman flavor — a sub-optimal solution always exists).
4. Trapezoidal (slab) decomposition (Latombe 1991; Preparata & Shamos 1985) is the popular exact decomposition: sweep a vertical line (slice) L->R through polygonal env; at each polygon VERTEX (event) the slice hits an IN/OUT/MIDDLE event; cells are trapezoids. PROBLEM: too many cells (every vertex spawns a cell), so too many redundant lengthwise back-and-forth motions. Fig 4: two trapezoids each 2.5x robot width need 3 passes each = 6; merged into one monotone cell needs 5. Fewer cells is better. Also requires polygonal env.
5. BCD = generalization/enhancement of trapezoidal: only OPEN/CLOSE cells at IN and OUT events — the events where the slice's CONNECTIVITY changes (number of connected free segments of the slice). MIDDLE events (vertex but connectivity unchanged) just UPDATE the current cell instead of splitting. -> All cells between an IN and an OUT merge into ONE cell (a monotone polygon), boustrophedon-coverable. Fewer cells -> fewer lengthwise passes.
6. The point where slice connectivity changes = a CRITICAL POINT (term from roadmap motion planning: Canny 1988 Roadmap; Canny & Lin 1990,1993 Opportunistic Path Planner). Because BCD keys on connectivity changes (not polygon vertices), it extends to curved / even sampled environments — not just polygons. (Morse-theory view in survey: cells defined by critical points of a slice/Morse function; later Acar & Choset sensor-based, Cao et al. convex-only.)
7. Events:
   - IN event: connectivity increases by 1 (slice was 1 segment, splits to 2). Old cell closes; two new cells open. Fig 2/8.
   - OUT event: connectivity decreases by 1 (2 segments merge to 1). Two cells close; one new cell opens. Fig 3/9.
   - MIDDLE: connectivity unchanged; update current cell, no split.
   - First event when slice enters free space from -inf: first cell opens. Terminates when slice leaves free space.
8. Adjacency graph built during sweep: each cell a node, edge between adjacent cells (segments overlapping across consecutive slices). A DEPTH-FIRST-like graph search outputs a path list = exhaustive walk visiting every node at least once.
9. Coverage execution from the path list: enter an "uncleaned" cell -> plan boustrophedon motion, then plan a path to next cell in list. Enter a "cleaned" cell -> just transit to next cell. Repeat until path list exhausted -> every cell cleaned -> env covered.

## Why each choice (design-decision -> why)
- Decompose at connectivity changes, NOT every vertex (trapezoidal): minimizes # cells hence # redundant lengthwise motions (turns are expensive: decelerate/turn/accelerate, and accrue dead-reckoning error). Merging IN..OUT cells = monotone polygon still boustrophedon-coverable.
- Slice/connectivity criterion (vs polygon vertices): generalizes to non-polygonal/curved/sampled environments; ties to critical points of roadmap theory.
- DFS exhaustive walk (not optimal TSP): determining the OPTIMAL coverage path is NP-complete (optimal even harder; may be intractable even under restrictive assumptions). So fall back to a complete-but-suboptimal DFS walk; user can vary sweep direction & re-score by a metric (path length) to converge near-optimal cheaply.
- Footprint-spaced passes: side-step = footprint width guarantees full coverage; shorter side-step reduces missed acute-angle gaps near obstacle boundaries at a time/coverage tradeoff.
- Discretization caveat (real-robot): acute-angle obstacle corners get skipped because of the discrete side-step; a final obstacle-following pass (wall-follow) patches the boundary — consistent with floor-cleaning needing a separate boundary mechanism.

## Empirical / motivating facts (-> context.md Background; NEVER fabricate)
- Trapezoidal-decomposition redundancy (Fig 4): two trapezoids each ~2.5 robot-widths -> 3 passes each = 6 lengthwise motions; one merged monotone cell -> 5. (Derivable/illustrative, in primary.)
- Turning is costly (decelerate-turn-accelerate) and induces dead-reckoning error; minimizing turns improves time-to-completion (survey).
- Determining an optimal coverage path is NP-complete (survey/primary conclusion).
- Real-robot observation (primary experiments — these are about the METHOD's own runs; treat carefully): boustrophedon skips acute-angle regions near obstacle boundaries due to discrete side-step; dead-reckoning error accumulates and can cause crashes -> sensor feedback needed. These are diagnostic observations about discretization/dead-reckoning (properties of the world/robot), used as motivation for the wall-follow patch and sensor-based future work; I will phrase as known robot/discretization facts, not as proposed-method benchmark wins. I will NOT cite coverage-percentage numbers (none given).
- DO NOT invent any coverage % or runtime numbers — none are in the sources.

## Canonical code structure (to ground final code, from PythonRobotics + Karlish)
- Grid/raster representation: free=1, obstacle=0 (Karlish uses occupancy image).
- calc_connectivity(slice column) -> (#connected segments, list of (start,end) intervals). The slice = one grid column; connectivity = # maximal runs of free cells.
- get_adjacency_matrix(parts_left, parts_right): overlap test min(l[1],r[1]) - max(l[0],r[0]) > 0 -> two segments are vertically connected across consecutive columns.
- bcd main loop over columns: last_connectivity==0 -> open #connectivity cells; connectivity==0 -> close all; else use adjacency matrix: a left part connected to >1 right parts = IN (open new cells), a right part connected to >1 left = OUT, a right part connected to 0 left = new isolated cell. Assign cell ids per slice segment; record boundaries.
- adjacency graph from cell overlaps; DFS walk over cells.
- per-cell boustrophedon: for each x-column in cell, sweep y up or down alternating (j%2), step by robot_size. (move_boustrophedon / PythonRobotics SweepSearcher with turning window + swap_moving_direction at obstacle.)
- PythonRobotics single-cell sweep: SweepSearcher.move_target_grid moves along moving_direction; on hitting occupied uses a "turning window" of candidate moves to step over by sweep_direction and swap_moving_direction (the back-and-forth flip). is_search_done when goal-row fully occupied (covered).

## In-frame rules
- NEVER name the target paper/authors/venue/method-as-paper. May name "boustrophedon cellular decomposition" as the thing being built (mainly answer.md).
- Prior-art citations (Latombe 1991, Preparata & Shamos 1985, Canny 1988, Canny & Lin 1990/1993, Zelinsky et al. 1993, Hert et al. 1996, Hofner & Schmidt 1995, Colegrave & Branch 1994, Ollis & Stentz 1996, VanderHeide & Rao 1995, Kurabayashi et al. 1996, Elfes/Moravec, Gabriely & Rimon STC, Hert & Lumelsky) STAY.
- Frontier anchor (UAV inspection/wind) — never mention.

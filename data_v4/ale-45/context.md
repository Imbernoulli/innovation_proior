# Watchman Route (max area seen, bounded steps)

## Research question

A guard patrols a grid floor-plan of `H x W` cells, some of which are walls. Standing on a
free cell the guard *sees* a set of cells by line-of-sight; walking from cell to cell the guard
accumulates everything ever seen. The guard must return to where it started, and it can take at
most `L` steps. How should the route be chosen so that the number of **distinct cells ever seen**
is as large as possible?

This is the discrete *watchman route* problem: instead of asking for the shortest closed walk that
sees the whole region (the classic watchman tour), we fix a **step budget** and maximize coverage.
With a tight budget the guard cannot look down every side pocket, so the route has to *select* which
parts of the floor-plan to inspect. That makes it a budgeted, submodular coverage / orienteering
problem on a grid graph: NP-hard, no closed-form optimum, judged by a continuous coverage count.

## Input / output contract

- **Input (stdin)** — one instance:
  - line 1: `H W L` — grid height, width, and the step budget `L`.
  - next `H` lines: `W` characters each, the grid. `'.'` = free cell, `'#'` = wall,
    `'S'` = the start cell (exactly one; the start is itself a free, walkable cell).
- **Output (stdout)** — one solution: a single token, the **move string**, a possibly-empty
  sequence over the alphabet `{U, D, L, R}`. Starting at `S`, each character moves the guard one
  cell: `U` = row-1, `D` = row+1, `L` = col-1, `R` = col+1. To denote the empty route (no moves)
  the solver may print a single `0` (or `-`); both mean "stay at the start".
- **Time limit:** ~2 seconds per instance. **Memory:** 256 MB.

The guard occupies a free cell at all times; a route is a sequence of cells beginning and ending at
`S`. The route is **closed** (it must return to `S`) and uses **at most `L`** moves.

## Background

Define **rook line-of-sight**: from a free cell the guard sees its own cell plus every free cell
reachable in a straight horizontal or vertical run that is not blocked by a wall — each ray stops at
the first wall. This is the standard grid analogue of line-of-sight visibility and it makes the
coverage of a *cell* a fixed set, and the coverage of a *route* the union of those sets over every
visited cell.

Two structural facts drive everything. First, route coverage is a **monotone submodular** function
of the visited-cell set: visiting more cells never loses coverage, and the marginal gain of adding a
cell shrinks as the route already covers more — exactly the diminishing-returns structure that
budgeted-coverage heuristics exploit, and that admits a submodular upper bound for pruning. Second,
the binding constraint is *visibility*, not *reachability*: a guard that merely snakes down a
corridor never turns to look into the rooms hanging off it, so a naive sweep that "walks everywhere
it can afford" leaves whole pockets unseen. The lever is therefore **which cells to stand on**, with
the route stitched together by grid shortest paths.

Two reference approaches frame the problem. The **boustrophedon sweep** baseline walks a depth-first
serpentine tour of the connected free region from the start (a closed Euler tour of a spanning tree),
cut off when the remaining budget can no longer afford the backtrack home — cheap and always closed,
but it spends its budget *moving* rather than *looking*, so its coverage badly trails the ceiling.
The **orienteering** view treats each cell the guard might stand on as a prize (its marginal newly
seen cells) and the walk between chosen cells as a cost (grid shortest-path length), and asks for the
best closed budgeted tour — which is the combinatorial heart.

## Evaluation settings

- **Score (per instance):** the number of distinct cells ever seen along the submitted route,
  computed by the frozen scorer `verify/score.py`, which: (1) parses the move string; (2) checks
  feasibility; (3) replays the route from `S`, collecting the visited cells; (4) unions the
  rook line-of-sight set of every visited cell and reports its size.
- **Feasibility -> 0 floor.** The score is forced to `0` if the output is malformed (missing, or
  more than one whitespace token), if the move string contains a character other than `U,D,L,R`
  (after the empty-route sentinels `0`/`-`), if its length **exceeds `L`** (over budget), if any
  move steps **off the grid or onto a wall** (walks into a wall), or if the route is **open** (does
  not return to the start cell at the end).
- **Instances** are produced by `verify/gen.py SEED`: a grid of `28..34` per side with `16%..22%`
  of interior cells carved into walls (border kept open; every wall placement is rejected if it
  would disconnect the free cells, so the free region is always connected). The start is the free
  cell nearest the top-left corner. The budget is **tight**, `L = round(freeN * U(0.25, 0.35))`
  (with a small floor), so a closed route of length `L` cannot stand near enough cells to see them
  all — selection is forced.
- **Normalization / reporting:** the raw distinct-seen count is reported and compared against the
  boustrophedon sweep baseline (`verify/baseline.py`) on the same seeds; a higher mean is better.
  The seed set used for self-verification is seeds `1..20`.

## Code framework

A single self-contained C++17 program that reads the instance and writes a feasible move string. The
scaffold below reads the grid and emits the empty (always-feasible) route; the heuristic goes where
marked.

```cpp
#include <bits/stdc++.h>
using namespace std;

static const int DR[4] = {-1, 1, 0, 0};   // U,D,L,R
static const int DC[4] = {0, 0, -1, 1};
static const char MV[4] = {'U', 'D', 'L', 'R'};

int main() {
    int H, W, L;
    if (!(cin >> H >> W >> L)) return 0;
    vector<string> grid(H);
    for (int r = 0; r < H; r++) cin >> grid[r];

    int SR = -1, SC = -1;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] == 'S') { SR = r; SC = c; }

    // role per cell: '#' wall, otherwise free; 'S' is the (free) start.
    // visibility(r,c) = rook line-of-sight: the cell itself plus every free cell
    // in an unobstructed straight run up/down/left/right (each ray stops at a wall).
    //
    // TODO: heuristic -- choose a CLOSED walk from S of <= L steps that maximizes
    // the number of distinct cells ever seen (union of visibility over visited cells).
    // Output a FEASIBLE move string: closed (ends at S), length <= L, never steps
    // off-grid or onto a wall.  The empty route "0" is always feasible.

    string moves;                 // sequence over {U,D,L,R}
    if (moves.empty()) { printf("0\n"); return 0; }
    printf("%s\n", moves.c_str());
    return 0;
}
```

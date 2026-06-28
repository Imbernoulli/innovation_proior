# Grid Light Placement

## Research question

A warehouse floor is given as an `H x W` grid of cells. Each cell is either **floor**
(`.`) or **wall** (`#`). You may install ceiling **lights**, but only on floor cells.
A light shines down its **row** and its **column**: it illuminates its own cell and
every cell reachable along the same row or the same column **without crossing a wall**
— in each of the four directions it stops just before the first `#` (or at the grid
edge). The walls cast shadows, so a single light lights exactly the maximal clear cross
through its cell.

Every floor cell must end up lit. The task is to **place as few lights as possible**
so that the whole floor is illuminated. This is a covering / location problem on a grid
with shadows; the joint decision of *which* cells to light is combinatorial (it is a
set-cover-flavoured optimization, NP-hard in general), so the benchmark scores a
submission by *how few* lights it uses rather than by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first line is `H W` (`1 <= H, W <= 50`). Then `H` lines follow,
  each a string of length `W` over the alphabet `{'.', '#'}`. `'.'` is a floor cell that
  must be lit; `'#'` is a wall (it blocks light and may not hold a light).
- **Output (stdout):** the first line is `K`, the number of lights placed. Then `K` lines
  follow, each `r c` (0-indexed, `0 <= r < H`, `0 <= c < W`), the position of one light.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff: (1) the output parses as `K` followed by exactly `K`
integer pairs in range and nothing else; (2) every listed light sits on a floor cell
`'.'` (never on a wall, never out of bounds); and (3) after casting light from every
listed source (each stopping at the first wall in each direction), **every floor cell is
lit**. Anything else — a light on a wall, a leftover dark floor cell, a wrong count, a
stray token, a missing file — is **infeasible**.

## Background

Place a light on a floor cell and ask which cells it lights. Along its row it reaches
exactly the **maximal horizontal run of floor cells** that contains it (bounded by walls
or the grid edge); along its column it reaches exactly the **maximal vertical run** that
contains it. Call these the cell's **H-corridor** and **V-corridor**. So:

- Every floor cell belongs to exactly one H-corridor and exactly one V-corridor.
- A light placed at a cell "activates" (covers) its whole H-corridor and its whole
  V-corridor.
- A floor cell is **lit** iff a light has been placed *somewhere in its H-corridor* **or**
  *somewhere in its V-corridor*.

That reframing is the whole problem: a light is the choice of one (H-corridor, V-corridor)
intersection, and lighting everything means every floor cell has at least one of its two
corridors covered. This is a **set-cover / covering** structure — choosing a few
intersections to cover all cells — which is the regime where the standard strong recipe is
**greedy maximum-coverage construction followed by a local-search metaheuristic**. Several
approaches sit on the table:

- **One light per horizontal corridor.** Drop a light anywhere in each maximal horizontal
  run. Every floor cell's H-corridor then has a light, so the floor is fully lit. Always
  feasible, dead simple, and it uses exactly `B` lights where `B` is the number of
  H-corridors — but it ignores that a single light also lights a whole column, so it is
  wasteful. This is the reference baseline.
- **Greedy maximum-coverage.** Repeatedly place the light that newly illuminates the most
  still-dark floor cells (a light's marginal value = dark cells in its H-corridor plus dark
  cells in its V-corridor). The classic greedy set-cover heuristic; a much stronger
  feasible start than one-per-corridor.
- **Simulated annealing on the light count.** From a feasible set, remove a light, repair
  any cells it left dark with the cheapest covering light, and accept the change by an SA
  rule on the number of lights. This escapes the greedy's local optima.

The decisive engineering lever is the **corridor decomposition with per-corridor light
counters**. Keep, for each H-corridor and V-corridor, how many placed lights lie inside it;
a cell is lit iff its H-corridor count or its V-corridor count is positive. Adding or
removing one light changes exactly two counters, so deciding which cells just went dark (or
lit) costs only the size of those two corridors — an **incremental `O(affected cells)`
evaluation**, never a full grid re-simulation. That is what makes the remove-and-repair
inner loop of the SA cheap enough to run hundreds of thousands of times.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`)
  deterministically chooses `H, W ∈ [30, 50]` and fills the grid with floor, then adds a
  sparse random scatter of single-cell walls ("pillars") plus a few axis-aligned wall
  "bars" (partial interior walls that do not span a whole side). The bars chop the floor
  into many maximal corridors, so the H-corridor / V-corridor choice genuinely matters. At
  least one floor cell is guaranteed.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted
  solution.
  - **Feasibility floor:** if the output does not parse as `K` + `K` valid in-range pairs,
    or any light is on a wall / out of bounds, or any floor cell is left dark after
    re-simulating illumination from every light, the score is **`0`**.
  - Otherwise let `K` be the number of lights used and let `B` be the number of maximal
    horizontal floor corridors (the one-light-per-horizontal-corridor reference, which is
    always feasible and uses `B` lights — the scorer recomputes `B` itself). The score is

    ```
    score = round( 1 000 000 × B / K )      (feasible, K >= 1)
    score = 0                               (infeasible)
    score = 1 000 000                       (degenerate: no floor cells, K = 0)
    ```

    A higher score is better. The one-per-horizontal-corridor reference scores exactly
    `1 000 000`; using fewer lights scores strictly more, using more scores less but stays
    positive.
- **Reported metric.** The mean score over a fixed seed set. A genuine corridor-set-cover
  solver should land well above `1 000 000` (≈ 1.8–2.7× on these instances, i.e. roughly
  half the lights of the reference); the reference is the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a
feasible `K` + light list to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) return 0;
    vector<string> G(H);
    for (int r = 0; r < H; r++) {
        char buf[1 << 16];
        if (scanf("%s", buf) == 1) G[r] = string(buf);
        if ((int)G[r].size() < W) G[r].append(W - G[r].size(), '.');
    }

    // A feasible answer is ANY set of floor-cell lights whose cast illumination
    // covers every '.'. The trivial always-feasible start is one light per
    // maximal HORIZONTAL corridor: that lights every cell via its row.
    vector<pair<int,int>> lights;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (G[r][c] == '.' && (c == 0 || G[r][c-1] != '.'))
                lights.push_back({r, c});   // start of a horizontal corridor

    // TODO heuristic: decompose the floor into maximal H- and V-corridors; keep
    // per-corridor light counts (a cell is lit iff its H- or V-corridor count is
    // positive); greedy maximum-coverage construction; then simulated annealing
    // that removes a light and repairs newly-dark cells with the cheapest
    // covering candidate, using the O(affected cells) incremental evaluation.
    // Keep `lights` a feasible covering at all times so any early stop is valid.

    string out = to_string((int)lights.size()) + "\n";
    for (auto &p : lights) out += to_string(p.first) + " " + to_string(p.second) + "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```

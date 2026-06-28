# Maze Treasure Collection (time-budgeted)

## Research question

An autonomous collector starts on a marked cell of a rectangular maze and may walk for at
most `T` unit steps, each step moving one cell up, down, left or right onto open floor (never
into a wall, never off the grid). Scattered across the open floor are `K` **treasures**, each a
positive value sitting on a particular cell; stepping onto a treasure cell collects its value
**once** (revisiting it later adds nothing). The task is to choose a walk — a string of moves —
that **maximizes the total value collected** before the step budget runs out.

Structurally this is **orienteering on a grid**: not "visit everything" (that is impossible
within `T`) and not "find a shortest path" (there is no fixed destination), but "given a travel
budget, pick which prizes are worth the detour and in what order to grab them". The order and the
selection are coupled — a high-value treasure far away may cost so many steps that two nearby
small ones would have been the better spend. Orienteering is NP-hard, there is no known efficient
exact optimum at this scale, and the benchmark scores a walk by *how much value it collects*
rather than by matching a unique answer.

## Input / output contract

- **Input (stdin):**
  - First line: three integers `H W T` — the maze has `H` rows and `W` columns
    (`21 ≤ H, W ≤ 41`) and the step budget is `T` (a few hundred up to ~1000).
  - Then `H` lines, each a string of exactly `W` characters over `{ '#', '.', 'S' }`:
    `#` is a wall, `.` is open floor, and `S` is the **unique start cell** (also open floor).
  - Then a line with one integer `K` (the number of treasures).
  - Then `K` lines, each `r c v`: a treasure of value `v ≥ 1` on the open cell at row `r`,
    column `c` (`0 ≤ r < H`, `0 ≤ c < W`). No treasure sits on the start cell, and treasure
    cells are open floor.
- **Output (stdout):** a single **move string** over the alphabet `{ U, D, L, R }`
  (`U`=row−1, `D`=row+1, `L`=col−1, `R`=col+1). Whitespace in the output is ignored; only the
  move characters matter. An empty move string (the collector stays on `S`) is a valid output.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff (i) the move string has length `≤ T`, and (ii) every move stays on
the grid and lands on open floor (never a `#`). Any other output — too many moves, a move into a
wall or off the grid, or a stray non-move token — is **infeasible**.

## Background

Collapsing the maze onto its **points of interest** (POIs = the start cell plus the `K` treasure
cells) is the move that makes everything else tractable. One breadth-first search from each POI
gives the exact grid shortest-path distance between every pair of POIs (walls and corridors
already baked in), turning the maze into a complete weighted graph on `K+1` nodes. The problem
becomes a clean **node-weighted orienteering instance**: starting at the start node, choose an
ordered subset of treasure nodes whose summed hop distances are `≤ T`, maximizing summed value.
The grid disappears; only a `(K+1)×(K+1)` distance matrix and the values remain. (The BFS parent
pointers are kept so any chosen POI order can be stitched back into a concrete `U/D/L/R` string.)

Several approaches sit on the table once the instance is in this graph form:

- **Greedy nearest-treasure.** Repeatedly walk to the closest still-uncollected treasure while
  budget remains. Cheap and surprisingly strong on dense mazes because it is extremely
  travel-efficient — but it ignores value entirely, so it can fill the budget with cheap nearby
  prizes and skip a cluster of valuable ones. This is the deterministic reference the scorer
  normalizes against.
- **Value/cost greedy.** Insert treasures by best value-per-added-distance. Captures value the
  nearest-only greedy misses, but on its own gets stuck in poor local arrangements.
- **Beam search over partial routes.** Keep a frontier of the most promising partial walks,
  expand each by appending a few good next treasures, and **rank by an admissible upper bound**
  (current value + value still reachable within the leftover budget). The bound lets the beam
  prune dominated routes cheaply and find high-value backbones the greedy seeds never reach.
- **Local search / iterated local search.** Once a route exists, **2-opt** reversals shorten it
  (freeing budget), **cheapest-insertion** spends the freed budget on more treasures, and a
  **double-bridge** kick escapes local optima — the established strong, simple metaheuristic for
  budgeted routing at this scale.

The decisive engineering lever is **incremental evaluation on the POI graph**: a 2-opt reversal
or an insertion changes only a handful of edges, so its effect on the route's travel cost is an
`O(1)` delta over exactly those edges — the route length is never recomputed from scratch — and
the value delta is just the value of the node added or removed. That, plus the admissible bound in
the beam, is what makes thousands of candidate routes per instance affordable inside the budget.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  builds a maze by randomized-DFS carving on an odd lattice (a connected, corridor-rich maze with
  guaranteed reachability), then knocks down a small fraction of interior walls to create loops and
  shortcuts (so the routing choice is genuinely non-trivial). It picks a random open start cell, a
  step budget `T` set to a fraction of the open-cell count (enough to roam but far short of
  covering everything), and `K` treasures on open cells with **skewed values** (mostly small, a few
  large) — exactly the regime where which prizes you chase matters.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted move
  string.
  - **Feasibility floor:** if the move string has length `> T`, or any move leaves the grid or
    lands on a wall, or contains a stray non-move token, the score is **`0`**.
  - Otherwise let `c` be the total value of the **distinct** treasure cells the walk ever occupies
    (start counts, each treasure once), and let `B` be the value collected by the scorer's own
    deterministic greedy nearest-treasure baseline (repeatedly BFS to the nearest uncollected
    treasure and walk there while budget allows), recomputed inside the scorer so the reference is
    reproducible and independent of the solver. The score is

    ```
    score = round( 1 000 000 × c / B )      (feasible, B > 0)
    score = 1 000 000                        (feasible and B = 0, i.e. nothing reachable)
    score = 0                                (infeasible)
    ```

    A higher score is better. The greedy baseline scores exactly `1 000 000`; collecting more value
    scores strictly more; collecting less scores less but stays positive. The trivial *empty walk*
    collects nothing and scores `0` — the floor to beat.
- **Reported metric.** The mean score over a fixed seed set. A genuine orienteering solver should
  land well above `1 000 000` (≈ 1.10–1.40× on these instances) by out-collecting the
  nearest-treasure greedy.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible move
string to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, T;
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    vector<string> grid(H);
    for (int r = 0; r < H; r++) { char buf[200005]; scanf("%s", buf); grid[r] = buf; }
    int K; scanf("%d", &K);
    vector<int> tr(K), tc(K), tv(K);
    for (int i = 0; i < K; i++) scanf("%d %d %d", &tr[i], &tc[i], &tv[i]);

    // The empty walk (stay on S) is ALWAYS feasible: it collects nothing but
    // never steps into a wall and uses 0 <= T moves. Start from it so we always
    // have a legal answer in hand.
    string moves = "";

    // TODO heuristic: collapse the maze onto POIs (start + treasures), BFS from
    // each POI for the exact pairwise distance matrix (keeping parent pointers),
    // solve the resulting node-weighted ORIENTEERING instance (greedy seed +
    // beam search with an admissible bound + 2-opt/insertion iterated local
    // search under a ~1.85s budget), then stitch the chosen POI order back into
    // a U/D/L/R string. Keep `moves` feasible (length <= T, never into a wall).

    fputs(moves.c_str(), stdout);
    fputc('\n', stdout);
    return 0;
}
```

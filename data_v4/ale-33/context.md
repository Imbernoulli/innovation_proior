# Grid Wire Routing (disjoint rectilinear paths, minimum total length)

## Research question

You are given an `H x W` grid of cells and `n` **terminal pairs**. Pair `k` has two distinct cells,
endpoint 1 `(r1, c1)` and endpoint 2 `(r2, c2)`. You must connect every pair by a **wire**: a
4-connected rectilinear path, i.e. a sequence of grid cells in which consecutive cells differ by
exactly one in a single coordinate (a unit step up/down/left/right), starting at endpoint 1 and
ending at endpoint 2. The hard constraint is that the wires must be **vertex-disjoint**: no grid
cell may be used by more than one wire (endpoints included). Among all such disjoint routings you
want the one of **minimum total length**, where length is the number of edges summed over all wires.

This is a grid version of the classic *minimum-cost disjoint paths* / *global routing* problem from
VLSI. Deciding whether `n` vertex-disjoint paths even exist on a grid is NP-hard, and minimising
their total length is harder still; there is no efficient exact solver, so the quality of a heuristic
is judged on a continuous score. A routing in which any wire is broken, or any cell is shared by two
wires, is **infeasible** and scores `0`.

## Input / output contract

The solver reads one instance from **stdin** and writes one solution to **stdout**.

**Input (instance):**
```
H W n
r1 c1 r2 c2        <- pair 0
r1 c1 r2 c2        <- pair 1
...
r1 c1 r2 c2        <- pair n-1
```
All coordinates are 0-based (`0 <= r < H`, `0 <= c < W`). Every endpoint cell is distinct (no two
endpoints, across all pairs, coincide), and a pair's two endpoints differ. Sizes are moderate
(`H, W` around 18-30; `n` around 6 to a few dozen, scaled so the grid is congested but routable).

**Output (solution):** for each pair, in input order, one path record on its own line:
```
L  r_0 c_0  r_1 c_1  ...  r_{L-1} c_{L-1}
```
`L` is the number of cells the wire visits; then the `L` cells in order. The first cell must equal
endpoint 1 of that pair and the last must equal endpoint 2. The wire's edge length is `L - 1`.
Whitespace (spaces or newlines) separates all tokens.

Time limit: a few seconds per instance (the reference solver finishes in well under three seconds).

## Background

Two ideas frame the problem before committing to one.

- **Sequential shortest-path routing (rip-up-free greedy).** Route the pairs one at a time. For each
  pair, run a BFS/Dijkstra shortest path over the cells **not yet claimed** by an earlier wire, then
  permanently block the cells it uses. This is `O(n * H * W)` and trivial, but it is **order
  dependent and incomplete**: an early wire can take a corridor that boxes a later pair in, leaving
  it with no free path at all. When that happens the greedy has no valid output for that pair, the
  routing is infeasible, and the whole instance scores `0`. On congested instances this failure is
  the rule, not the exception.

- **Negotiated-congestion rip-up-and-reroute (PathFinder).** The established VLSI global-routing
  heuristic. Instead of hard-blocking, every cell carries a *cost* that rises as wires contend for
  it. All wires are ripped up and rerouted each pass on the current cost field; a **present-sharing
  penalty** makes a cell expensive while several wires sit on it, and a **historical-congestion
  term** accumulates on chronically contested cells so wires permanently learn to detour around the
  hot spots. Over passes the contention is *negotiated away* and the routing becomes disjoint, while
  staying near shortest length because the base per-cell cost is `1`, so uncontested detours stay
  cheap.

The non-obvious lever this datapoint targets is the second family: **negotiated-congestion
rip-up-and-reroute**, backed by a deterministic always-terminating sequential completer so a feasible
disjoint routing is *guaranteed*, plus a length-shrinking large-neighbourhood rip-up that uses the
remaining time budget to pull the total length down toward the unconstrained shortest-path bound.

## Evaluation settings

A deterministic scorer reads the instance and the solution and applies this rule:

1. **Parse + path validity.** Read `n` path records. The score is `0` if parsing fails, if any cell
   is outside the grid, if within a path two consecutive cells are not a unit rectilinear step (no
   diagonals, jumps, or repeated cell within a path), or if a path's endpoints do not match its
   pair (first cell = endpoint 1, last cell = endpoint 2).
2. **Disjointness (feasibility floor).** If **any** grid cell is used by more than one wire, the
   score is `0`. This is the feasibility -> 0 floor: a single shared cell, or a single broken wire,
   voids the entire output.
3. **Normalised score.** If feasible, let `total` be the total wire length (sum of `L_k - 1`) and
   let `LB` be the sum of the pairs' **Manhattan distances** — the length each pair would have if
   routed independently as an unconstrained shortest path, ignoring all conflicts (so conflicts
   effectively "count as failures" because conflicting routings never reach this branch). The score
   is

   ```
   score = LB / total.
   ```

   Because a disjoint routing can only be longer than the unconstrained shortest paths, `total >= LB`,
   so `score <= 1`; higher (closer to `1`) is better. The trivial sequential greedy scores `0` on any
   instance where it fails to route every pair.

**Instance generation.** A generator seeded by an integer builds each instance so that a disjoint
routing is **guaranteed to exist**: it first carves `n` vertex-disjoint rectilinear paths into an
empty grid with randomized self-avoiding walks, then hands the solver only the endpoints of those
carved paths. Because a disjoint routing demonstrably exists (the carved one), the instance is always
routable and the feasibility floor is meaningful. Grid sizes and the pair count vary with the seed so
the seed set spans easy (sparse) to congested regimes. The generator, scorer, and seed set are
frozen.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible solution
to stdout. The pre-method scaffold below already prints a guaranteed-feasible answer when one is easy
to find (sequential shortest-path routing); the heuristic replaces the `// TODO` block with
negotiated-congestion rip-up-and-reroute plus the length-shrinking local search.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, n;
    if (scanf("%d %d %d", &H, &W, &n) != 3) return 0;
    struct Pair { int sr, sc, tr, tc; };
    vector<Pair> P(n);
    for (auto& p : P) scanf("%d %d %d %d", &p.sr, &p.sc, &p.tr, &p.tc);

    // TODO: route every pair by a 4-connected rectilinear path so that no grid
    //       cell is used by more than one wire, minimising the total length.
    //       A safe starting point: route pairs one by one by BFS over the cells
    //       not yet used by an earlier wire (sequential greedy).  The heuristic
    //       upgrades this to negotiated-congestion rip-up-and-reroute so the
    //       routing is disjoint even when greedy would box a pair in.
    vector<vector<pair<int,int>>> route(n);
    // ... fill route[k] with the cell list for pair k (endpoint1 .. endpoint2) ...

    for (int k = 0; k < n; k++) {
        printf("%d", (int)route[k].size());
        for (auto& cell : route[k]) printf(" %d %d", cell.first, cell.second);
        printf("\n");
    }
    return 0;
}
```

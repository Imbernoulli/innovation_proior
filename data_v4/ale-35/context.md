# Flood-Control Levee Placement

## Research question

A river valley is given as an `H x W` grid of cells, each carrying an integer
**height** `h[r][c]`. Some cells are flood **sources** (a breached dam, a storm
drain, a coastline). Water spreads by a deterministic rule: starting from the
source cells, water moves from an already-flooded cell `u` to a 4-adjacent
neighbour `v` whenever `h[u] >= h[v]` — that is, water flows **downhill or across
level ground, but never strictly uphill**. Left alone, the flood fills every cell
it can reach this way.

We are allowed to build at most `B` **unit levees**, one per cell. A levee makes
its cell **impassable**: water never enters a leveed cell and water cannot flow
*through* it (a levee deletes all flood edges touching that cell). Levees may not
be placed on a source cell. The task is to choose a set of at most `B` levee cells
to **minimize the number of flooded cells** after the flood settles (leveed cells
are dry barriers and do not count as flooded; source cells always count).

This is a min-cut-flavoured placement problem on an implicit directed flood graph:
the water spreads along directed edges `u -> v` (`h[u] >= h[v]`), and a budget of
`B` node deletions has to disconnect as much of the reachable region from the
sources as possible. It is **NP-hard** in general (a budgeted node multiway cut),
with no known efficient exact solver at this scale, and the benchmark scores a
levee set by *how few cells it leaves flooded*, not by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first line is `H W B S` (`28 <= H, W <= 40`,
  `4 <= B <= 16`, `1 <= S <= 3`). Then `H` lines follow, line `r` holding `W`
  integers `h[r][0] ... h[r][W-1]` (`0 <= h <= 100`), the height grid in row-major
  order. Then `S` lines follow, each `sr sc` — the (row, col) of a source cell
  (`0 <= sr < H`, `0 <= sc < W`).
- **Output (stdout):** first an integer `L` (`0 <= L <= B`), the number of levees
  placed; then `L` lines, each `r c` — the (row, col) of one levee cell. `L = 0`
  (build nothing) is a legal output.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff (a) the output parses as `L` then `L` valid `r c`
pairs; (b) `L <= B`; (c) every levee cell is inside the grid; (d) no levee
coincides with a source cell; and (e) no two levees occupy the same cell. Anything
else — a parse error, over budget, an out-of-grid levee, a levee on a source, a
duplicate — is **infeasible**.

## Background

The flood is a plain multi-source reachability over the directed graph whose edges
are `u -> v` for 4-adjacent `u, v` with `h[u] >= h[v]`. One BFS/DFS from the
sources reveals exactly which cells flood and how much water there is to fight.
The hard part is *where* to spend the budget. Several approaches sit on the table
before committing:

- **Do nothing (the baseline).** Place zero levees. Always feasible; the whole
  reachable region floods. It is the reference the scorer recomputes and the floor
  a real solver must beat.
- **Random / spread-out levees.** Drop `B` levees on random flooded cells. Feasible
  and trivial, but on terrain with real bottlenecks it barely helps: a levee in the
  middle of a wide flooded plain blocks one cell while the water flows around it.
- **Wall the source.** Ring each source with levees. This works only when a source
  has few outflow cells; when the source sits in a wide flat reservoir it has far
  too many escape routes to seal within the budget, so this is usually hopeless.
- **Cut the bottlenecks.** The decisive observation is that real terrain channels
  the flood through a few **narrow passes** (gaps in ridges) into large downstream
  **basins**. A single levee dropped in such a pass can keep an *entire* basin dry.
  The whole game is to find those high-value passes and spend the budget on them.

The lever that makes "cut the bottlenecks" both effective and cheap is using **one
flood pass to score every candidate levee's marginal benefit**:

- **Single-pass dominator estimate.** From one flood we build the flood BFS forest
  and, in one extra linear sweep, estimate each flooded cell's **downstream subtree
  size** — how many cells would lose their flooding if that cell were cut. A child
  `v` contributes its whole subtree to its parent only when `v` has a *unique*
  flooded in-neighbour (in-degree 1 in the flood graph), so the estimate counts a
  cell toward a candidate only when cutting that candidate truly disconnects it.
  This deliberately *under*-counts where alternate flood paths exist, which is
  exactly what ranks the genuine single-pass bottlenecks at the top — without
  re-flooding once per candidate. A greedy construction repeatedly floods, takes
  the top-scoring frontier cell as a levee, and repeats up to `B` times.
- **Exact re-flood for the polish.** Because the grid is small (`<= 40 x 40`), a
  full flood costs only `O(HW)`. So after the greedy seed, a simulated-annealing
  search over the levee *set* — add a levee, remove one, or move one — scores each
  candidate set by **one exact re-flood**, escaping the greedy's local optima while
  always keeping a feasible set in hand to print.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`)
  deterministically builds a valley with genuine bottleneck structure: a wide
  **flat reservoir plain** at a medium height holds the sources, enclosed by tall
  **ridges** each punched by one or two low **passes**; beyond every pass lies a
  large **basin** whose flat floor is lower than the plain. With no levees the
  plain floods fully, pours through every pass, and fills every basin — typically
  `> 90%` of the grid. The flat plain is too wide to wall off within the budget, so
  the only winning move is to plug the passes. The budget `B` is set a little above
  the number of passes, so the solver must pick the *right* passes rather than
  plug everything for free.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the
  submitted levee list.
  - **Feasibility floor:** if the output does not parse as `L` then `L` valid
    `r c` pairs, or `L > B`, or a levee is out of the grid, on a source, or a
    duplicate, the score is **`0`**.
  - Otherwise run the flood with the levees in place and let `flooded_solver` be
    the number of flooded cells (lower is better). Let `flooded_ref` be the flood
    count with **no levees** (the do-nothing baseline), recomputed inside the
    scorer so the reference is reproducible and solver-independent. The score is

    ```
    score = round( 1_000_000 * flooded_ref / flooded_solver )   (feasible, flooded_solver > 0)
    score = 0                                                     (infeasible)
    ```

    A higher score is better. The do-nothing baseline scores essentially
    `1_000_000`; a levee set that keeps cells dry scores strictly more; placing
    levees can never *increase* flooding (a levee only ever deletes flood edges),
    so a feasible solution never scores below the baseline. (The unreachable
    `flooded_solver == 0` case — impossible, since sources always flood and cannot
    be leveed — is given a full-credit cap rather than dividing by zero.)
- **Reported metric.** The mean score over a fixed seed set. A real
  bottleneck-cutting solver (single-pass greedy seed + re-flood SA) lands well
  above the floor — roughly `3x`–`4x` of `1_000_000` on these clustered-bottleneck
  instances — while the do-nothing baseline scores exactly `1_000_000` and random
  levees score only `~1.01x`, confirming the problem rewards finding the right
  passes, not placing levees anywhere.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing
a feasible levee list to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, B, S;
    if (scanf("%d %d %d %d", &H, &W, &B, &S) != 4) return 0;
    int N = H * W;
    vector<int> ht(N);
    for (int i = 0; i < N; i++) scanf("%d", &ht[i]);
    vector<char> isSrc(N, 0);
    vector<int> srcs;
    for (int i = 0; i < S; i++) {
        int sr, sc; scanf("%d %d", &sr, &sc);
        isSrc[sr * W + sc] = 1;
        srcs.push_back(sr * W + sc);
    }

    // A feasible answer is ANY set of at most B cells, none a source, no
    // duplicates -- e.g. the empty set (print "0"), which floods everything but
    // is always legal. Start from a legal set so there is always something to
    // print.
    vector<int> levees;

    // TODO heuristic: run ONE flood from the sources (water flows u->v iff
    // h[u] >= h[v]); from that single pass estimate each flooded cell's
    // downstream "subtree size" (a dominator-flavoured count that only credits a
    // cell when cutting it truly disconnects the subtree), and greedily take the
    // top-scoring frontier cell as a levee, repeating up to B times. Then polish
    // with a simulated annealing over the levee SET (add / remove / move a
    // levee), scoring each candidate set by ONE exact re-flood (the grid is
    // small), keeping the best feasible set seen. Never exceed B; never place a
    // levee on a source or a duplicate.

    // Emit the chosen levees.
    string out = to_string((int)levees.size()) + "\n";
    for (int u : levees) { out += to_string(u / W) + " " + to_string(u % W) + "\n"; }
    fputs(out.c_str(), stdout);
    return 0;
}
```

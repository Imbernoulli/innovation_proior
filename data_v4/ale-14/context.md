# Graph Coloring with Soft Conflicts

## Research question

You are given a weighted undirected graph on `n` vertices and a fixed budget of `k` colors. A
*coloring* assigns each vertex one of the `k` colors. An edge is a **conflict** if both of its
endpoints receive the same color; its penalty is the edge's weight. The cost of a coloring is the
**total weight of all conflicting (monochromatic) edges**, and the task is to choose the coloring
that makes that total as small as possible.

The graphs are built so that `k` colors are *not* enough to avoid every conflict — the chromatic
number exceeds `k` — so the best achievable cost is strictly positive. This is therefore not the
yes/no question "is the graph `k`-colorable?" but a **soft-conflict minimization**: you cannot make
all edges happy, so you must decide *which* conflicts (which weights) to keep. It is the weighted
`k`-partition / minimum-conflict graph-coloring problem, which is NP-hard; there is no known
efficient exact solver at this scale, and the benchmark scores a coloring by *how little* conflict
weight it leaves, not by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first line holds three integers `n m k`:
  - `n` (`1 ≤ n ≤ 700`) — number of vertices, labelled `0 … n-1`;
  - `m` — number of edges;
  - `k` (`1 ≤ k ≤ 6`) — number of available colors.
  Then `m` lines follow, each `u v w` with `0 ≤ u, v < n`, `u ≠ v`, and integer weight
  `1 ≤ w ≤ 1000`. The graph is undirected and simple (no self-loops, no duplicate undirected
  pairs).
- **Output (stdout):** `n` lines (or `n` whitespace-separated integers), line `i` holding the color
  `color[i] ∈ {0, …, k-1}` assigned to vertex `i`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff the output is exactly `n` integers, each in the range `[0, k-1]`.
Anything else — wrong count, an out-of-range color, a non-integer token, a missing file — is
**infeasible** and scores `0`.

## Background

The objective is a *partition* objective: the `k` colors split the vertices into `k` groups, and the
cost is the weight of edges that stay inside a group. Equivalently we want to push as much edge
weight as possible *across* the cut between color classes. Several approaches sit on the table before
committing to one:

- **Greedy / DSATUR construction.** Order the vertices (e.g. by descending weighted degree) and give
  each vertex the color that adds the least conflict weight with its already-colored neighbours.
  This is `O(m + n·k)`, always feasible, and a reasonable start, but it commits early to choices it
  cannot revisit and typically leaves substantial weight on conflicting edges.
- **Local search by single recolors.** Repeatedly recolor one vertex to reduce the conflict cost.
  The natural and very strong member of this family for graph coloring is **TabuCol** (the
  Hertz–de Werra tabu search): at each step, recolor one *conflicting* vertex to its best alternative
  color, and forbid (make *tabu*) undoing that change for a tuned number of iterations so the search
  can climb off plateaus instead of cycling.
- **Metaheuristics on top.** Population methods (hybrid evolutionary coloring), simulated annealing,
  or partial-col variants exist, but for the minimum-*weighted*-conflict objective at this size,
  tabu search over recolors with a good move evaluator is the established strong, simple workhorse.

The decisive engineering lever — and the reason a naive recolor search is too slow — is **incremental
move evaluation**. Recomputing the whole conflict cost to score each of the `O(n·k)` candidate
recolors per step would be `O(n·k·m)`, hopeless. Instead, maintain a conflict-count table
`gamma[v][c]` = the weighted number of conflicts vertex `v` would have if it took color `c` (the sum
of weights of `v`'s neighbours currently colored `c`). Then the cost change of recoloring `v` from
its current color to `c` is exactly `gamma[v][c] − gamma[v][current]` — an `O(1)` lookup. Applying a
recolor of `v` updates `gamma` only for `v`'s neighbours (`gamma[u][old] −= w`, `gamma[u][new] += w`),
which is `O(degree(v))`, not `O(n)` or `O(m)`. This incremental table is what makes thousands of
tabu iterations per second feasible.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [400, 700]`, `k ∈ [3, 6]`, and builds a graph as a union of (a) several **planted
  dense clusters**, each of size `> k` and drawn from the whole vertex set with overlap, so each
  cluster's near-clique core *forces* leftover conflicts under `k` colors (pigeonhole), plus (b) a
  sparser **Erdős–Rényi background** coupling the clusters. Edge weights are **heavy-tailed**
  (mostly small, a few very large), so the optimizer must care *which* conflicts it keeps, not merely
  how many. This is the regime where a greedy coloring leaves the most weight on the table for local
  search to recover.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted coloring.
  - **Feasibility floor:** if the output is not exactly `n` integers each in `[0, k-1]`, the score is
    **`0`**.
  - Otherwise let `L` be the submitted coloring's total conflict weight (sum of `w` over edges whose
    two endpoints share a color), and let `G` be the conflict weight of the scorer's own
    deterministic **DSATUR-style greedy** baseline (vertices in descending weighted-degree order,
    each given the conflict-minimizing color; the scorer recomputes `G` itself, so the reference is
    reproducible and independent of the solver). The score is

    ```
    score = round( 1 000 000 × (G + 1) / (L + 1) )   (feasible)
    score = 0                                         (infeasible)
    ```

    Lower conflict weight `L` ⇒ higher score; the `+1` smoothing keeps the score finite even when a
    coloring reaches `L = 0`. The greedy baseline scores exactly `1 000 000` against itself; a
    lower-conflict coloring scores strictly more; a worse one scores less but stays positive.
- **Reported metric.** The mean score over a fixed seed set. A genuine tabu-search solver lands well
  above `1 000 000` (it consistently drives `L` well below the greedy `G`); the trivial *all-one-color*
  coloring (every vertex color 0) makes *every* edge a conflict and scores only a few tens of
  thousands — the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible coloring
to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, k;
    if (scanf("%d %d %d", &n, &m, &k) != 3) return 0;
    vector<int> eu(m), ev(m); vector<long long> ew(m);
    for (int i = 0; i < m; i++) scanf("%d %d %lld", &eu[i], &ev[i], &ew[i]);

    // A feasible answer is ANY assignment of colors in [0, k-1]. Start from
    // all-zero so we always hold something legal to print.
    vector<int> color(n, 0);

    // TODO heuristic: greedy (DSATUR-style) construction for a strong feasible
    // start; then TabuCol -- tabu search recoloring one conflicting vertex at a
    // time, using an incremental conflict-count table gamma[v][c] so each move's
    // cost delta is O(1) and applying a move is O(degree); tabu tenure tuned to
    // escape plateaus, with aspiration; all under a ~2s wall-clock budget. Keep
    // `color` valid (every entry in [0, k-1]) throughout.

    string out;
    for (int i = 0; i < n; i++) out += to_string(color[i]) + "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```

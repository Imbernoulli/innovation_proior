# Graph Coloring (Minimize the Number of Colors)

## Research question

You are given a simple undirected graph on `n` vertices. A *proper coloring* assigns each vertex one
color so that **no edge is monochromatic** — the two endpoints of every edge get different colors. The
task is to produce a proper coloring that uses **as few distinct colors as possible**.

The minimum number of colors over all proper colorings is the graph's **chromatic number** `χ(G)`.
Deciding whether `χ(G) ≤ k` is NP-complete, and computing `χ(G)` exactly is NP-hard; at the scale
here (`n` in the hundreds, thousands of edges) there is no known efficient exact solver. So this is a
heuristic optimization problem: any proper coloring is a valid answer, and a coloring is *better* the
fewer colors it spends. An **improper** coloring (any monochromatic edge) is not a solution at all and
scores zero. Graph coloring is the canonical model for register allocation, frequency assignment,
exam timetabling, and conflict-free scheduling, where "one color" is "one reusable resource" and fewer
colors means fewer resources.

## Input / output contract

- **Input (stdin):** the first line holds two integers `n m`:
  - `n` (`1 ≤ n ≤ 600`) — number of vertices, labelled `0 … n-1`;
  - `m` — number of edges.
  Then `m` lines follow, each `u v` with `0 ≤ u, v < n`, `u ≠ v`. The graph is undirected and simple
  (no self-loops, no duplicate undirected pairs).
- **Output (stdout):** `n` integers — `color[i] ≥ 0` for vertex `i` — as `n` whitespace-separated
  tokens (one per line is fine). The colors are arbitrary non-negative integer *labels*; only the
  **number of distinct labels used** matters, and they need not form a contiguous range.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff (a) the output is exactly `n` non-negative integers, and (b) the
coloring is **proper** — for every edge `(u, v)`, `color[u] ≠ color[v]`. Anything else — wrong token
count, a negative or non-integer token, a missing file, or **any monochromatic edge** — is
**infeasible** and scores `0`.

Example: for the triangle `n=3`, edges `{(0,1),(1,2),(0,2)}`, any assignment of three distinct colors
(e.g. `0 1 2`) is proper and optimal (`χ = 3`); the assignment `0 1 0` is improper (edge `(0,2)`
monochromatic) and scores `0`.

## Background

The objective — minimize the palette of a proper coloring — is a *covering by independent sets*: each
color class is an independent set, and we want to cover all `n` vertices with the fewest independent
sets. Several approaches sit on the table before committing to one:

- **First-fit / largest-first greedy.** Order the vertices (e.g. by descending degree) and give each
  the smallest color not already used by an already-colored neighbour. This is `O(n + m)`, always
  proper, and trivial to write — but its color count depends heavily on the order, and on dense
  graphs it tends to *open extra colors*: it repeatedly meets a vertex that already conflicts with
  every open color and starts a fresh one, ending several colors above `χ`.
- **DSATUR construction.** Greedy with a smarter order: always color next the uncolored vertex with
  the highest *saturation degree* (the number of distinct colors already present among its
  neighbours), ties broken by uncolored degree. DSATUR adapts the order to the partial coloring and is
  a markedly stronger constructive heuristic than fixed-order first-fit, often landing at or very near
  `χ` on structured graphs.
- **Local search to remove colors (the established strong method).** A single construction commits to
  choices it cannot revisit. The standard way to push the color count *below* what construction gives
  is to attack the *decision* problem repeatedly: fix a target `k` (one less than the current best),
  and search for a **proper `k`-coloring** by local search that minimizes the number of conflicting
  edges down to zero. The canonical local search for this is **TabuCol** (Hertz–de Werra): repeatedly
  recolor one *conflicting* vertex to its best alternative color, and forbid (make *tabu*) undoing
  that change for a tuned number of iterations so the search climbs off plateaus instead of cycling.
  If a proper `k`-coloring is found, lower `k` and repeat; when `k` becomes unreachable, the last
  proper coloring is the answer. (Kempe-chain interchanges are a classic alternative local move for
  the same goal.)

The decisive engineering lever — and the reason a naive recolor search is too slow — is **incremental
move evaluation**. Recomputing the whole conflict count to score each of the `O(n·k)` candidate
recolors per step would be `O(n·k·m)`, hopeless. Instead, maintain a conflict-count table
`gamma[v][c]` = the number of `v`'s neighbours currently colored `c`. The change in the number of
conflicts from recoloring `v` from its current color to `c` is exactly `gamma[v][c] − gamma[v][cur]`,
an `O(1)` lookup. Applying a recolor of `v` updates `gamma` only for `v`'s neighbours
(`gamma[u][old] −= 1`, `gamma[u][new] += 1`), which is `O(degree(v))`, not `O(n)` or `O(m)`. This
incremental table is what makes the thousands of tabu iterations per second — and therefore the
repeated `k`-feasibility searches — affordable inside a two-second budget.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [350, 550]` and a hidden class count `C ∈ [8, 14]`, then **plants a proper
  `C`-coloring**: it partitions the vertices into `C` random classes and adds edges **only between
  different classes** (never inside a class). By construction the planted partition is proper, so
  `χ(G) ≤ C`. Inter-class edges are added densely (a high pairwise probability), then thinned to a
  target average degree, producing a near-complete-multipartite core (so `χ` is also pinned *close*
  to `C` from below) over a high-degree graph. This is precisely the regime where a plain first-fit
  greedy *wastes* colors — it opens several more than needed — and a DSATUR + tabu local search can
  recover them. The class assignment is random, so the structure cannot be read off the vertex
  indices.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted coloring.
  - **Feasibility floor:** if the output is not exactly `n` non-negative integers, **or** the coloring
    is **improper** (some edge is monochromatic), the score is **`0`**.
  - Otherwise let `K` be the number of **distinct colors** the submitted coloring actually uses, and
    let `G` be the number of colors used by the scorer's own deterministic **first-fit greedy**
    baseline (vertices in descending-degree order, each given the smallest color free among colored
    neighbours; the scorer recomputes `G` itself, so the reference is reproducible and independent of
    the solver). The score is

    ```
    score = round( 1 000 000 × G / K )   (feasible, K ≥ 1)
    score = 0                            (infeasible)
    ```

    Fewer colors `K` ⇒ higher score. Using strictly fewer colors than greedy (`K < G`) scores **above**
    `1 000 000`; matching greedy scores exactly `1 000 000`; using more scores below it but stays
    positive. (For the degenerate `n = 0` instance the score is `1 000 000`.)
- **Reported metric.** The mean score over a fixed seed set (seeds `1 … 20`). A genuine DSATUR + tabu
  solver lands well above `1 000 000` (it consistently colors the graph with 2–4 fewer colors than
  greedy); the trivial *all-distinct* coloring (every vertex its own color, `K = n`) is proper but
  spends `n` colors and scores only a few tens of thousands — the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible coloring
to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        edges[i] = {u, v};
    }

    // A feasible answer is ANY proper coloring. The all-distinct coloring
    // (color[i] = i) is always proper, so we can always hold something legal.
    vector<int> color(n);
    for (int i = 0; i < n; i++) color[i] = i;

    // TODO heuristic: DSATUR construction for a strong proper start (k0 colors);
    // then color reduction -- repeatedly target k = k0-1, k0-2, ... by TabuCol,
    // a tabu search that recolors one conflicting vertex at a time using an
    // incremental conflict table gamma[v][c] (O(1) move delta, O(degree) update)
    // to drive the conflict count to 0; on success adopt the proper k-coloring
    // and lower k, else stop. Keep `color` a valid proper coloring throughout,
    // all under a ~2s wall-clock budget.

    string out;
    for (int i = 0; i < n; i++) out += to_string(color[i]) + "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```

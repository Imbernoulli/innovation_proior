# Colorful forest: largest edge set that is both a forest and color-bounded

## Research question

You are given an undirected graph with `n` vertices and `m` edges. Every edge carries a
**color** in `{1, ..., K}`, and each color `c` has an integer **capacity** `cap[c] >= 0`.
You must choose a subset `S` of the edges that is simultaneously:

- a **forest** — the chosen edges contain no cycle (parallel edges and self-loops therefore
  cannot both/ever be kept if they close a cycle); and
- **color-bounded** — for every color `c`, at most `cap[c]` chosen edges have color `c`.

Output the **maximum possible size** `|S|`.

This is exactly *maximum-cardinality intersection of two matroids* on the common ground set of
the `m` edges: the **graphic matroid** (independent = forest) and a **partition matroid**
(independent = at most `cap[c]` elements per color class). The two constraints pull against each
other — the cheapest-looking edge to add for the forest may be the one whose color is already
exhausted — and a wrong but tempting "greedy by some local score" can leave size on the table.

## Input / output contract

- Input (stdin):
  - Line 1: three integers `n m K` (`1 <= n <= 2000`, `0 <= m <= 2000`, `1 <= K <= 2000`).
  - Line 2: `K` integers `cap[1] ... cap[K]` (`0 <= cap[c] <= m`).
  - Next `m` lines: three integers `u v c` describing an edge between vertices `u` and `v`
    (`1 <= u, v <= n`) with color `c` (`1 <= c <= K`). Self-loops (`u == v`) and parallel edges
    may appear.
- Output (stdout): a single line with the maximum size of a feasible edge subset.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with the graph below the answer is `3`.

```
4 5 2
2 1
1 2 1
2 3 1
3 4 2
4 1 2
1 3 1
```

Vertices `1..4`; colors `1,2` with capacities `2` and `1`. Edges (0-indexed): `e0=(1,2,c1)`,
`e1=(2,3,c1)`, `e2=(3,4,c2)`, `e3=(4,1,c2)`, `e4=(1,3,c1)`. A spanning tree of the 4 vertices has
3 edges; choosing `{e0, e1, e2}` uses color 1 twice and color 2 once, satisfying `cap=[2,1]`, so
`|S| = 3`. No feasible set has 4 edges (4 edges on 4 vertices must contain a cycle), so the answer
is `3`.

## Background

Two natural-but-flawed ideas should be confronted first:

- **Greedy by color scarcity / by anything local.** Sort edges by some heuristic (e.g. rarest
  color first, or pick any forest edge whose color still has room) and add greedily, skipping an
  edge if it would create a cycle or overflow a color. Greedy is `O(m alpha(n))` and trivial to
  write, but it commits irrevocably: an early edge can simultaneously occupy a forest "slot"
  (its tree path) and a color "slot", blocking a strictly better global combination. Greedy
  maximizes each single matroid, but the *intersection* of two matroids is not greedy-solvable.

- **Reduce to flow / matching.** A graphic matroid does not decompose into a bipartite matching,
  so the clean min-cut reductions that solve a single partition/transversal matroid do not apply
  to the forest constraint directly.

The exact, canonical tool is **matroid intersection via shortest augmenting paths in the exchange
graph**: start from the empty common independent set and repeatedly augment by one element along a
shortest source-to-sink alternating path; when no such path exists the current set is provably
maximum (Lawler / Edmonds). Each augmentation is driven entirely by the two matroids' independence
oracles, which here are a forest test (union-find / tree-path) and a per-color counter.

## Evaluation settings

Judged on hidden tests covering: empty edge set (`m = 0`); all capacities zero; a single color
with large capacity (reduces to "maximum spanning forest", answer = `n - #components`); many
colors each with capacity 1 (tight partition matroid); self-loops and parallel edges (which the
forest oracle must reject as cycle-closing); disconnected graphs; and dense instances at the
upper bounds where a naive exponential search is hopeless but the polynomial augmenting-path
algorithm finishes comfortably.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, K;
    if (scanf("%d %d %d", &n, &m, &K) != 3) return 0;
    vector<int> cap(K + 1, 0);
    for (int c = 1; c <= K; c++) scanf("%d", &cap[c]);
    vector<int> eu(m), ev(m), ec(m);
    for (int i = 0; i < m; i++) scanf("%d %d %d", &eu[i], &ev[i], &ec[i]);

    // TODO: compute the largest edge subset that is BOTH a forest (graphic matroid)
    // and respects every per-color capacity (partition matroid). This is maximum
    // matroid intersection.
    int answer = 0;

    printf("%d\n", answer);
    return 0;
}
```

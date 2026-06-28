# Dynamic connectivity under edge insertions and deletions (offline)

## Research question

You maintain an undirected graph on `n` vertices (numbered `1..n`) that starts with no edges.
You then process a timeline of `q` operations, in order. Each operation is one of:

- **add** an edge between `u` and `v` (the edge is guaranteed to be absent at that moment, and `u != v`);
- **remove** the edge between `u` and `v` (the edge is guaranteed to be present at that moment);
- **query** whether `u` and `v` are currently in the same connected component.

For every query you must output whether the two vertices are connected *given the edges that are
present at the time of that query*. The hard part is the **removal**: connectivity is easy to keep up
to date as long as edges only ever appear, but once edges can also disappear there is no incremental
component structure that supports a cheap "undo a merge". The question is how to answer all `q`
connectivity queries within tight limits when the edge set is fully dynamic.

## Input / output contract

- Input (stdin): the first line contains two integers `n` and `q`
  (`1 <= n <= 2*10^5`, `0 <= q <= 2*10^5`).
  Each of the next `q` lines contains three integers `type u v`:
  - `type = 1`: add edge `u`-`v` (guaranteed absent, `1 <= u, v <= n`, `u != v`);
  - `type = 2`: remove edge `u`-`v` (guaranteed currently present, `u != v`);
  - `type = 3`: query connectivity of `u` and `v` (`1 <= u, v <= n`; `u == v` is allowed and answers `YES`).
- Output (stdout): for each query (`type = 3`), in the order the queries appear, print `YES` if `u`
  and `v` are connected at that moment, otherwise `NO`. Print nothing for add/remove operations.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
Input
4 8
1 1 2
3 1 2
1 2 3
3 1 3
2 1 2
3 1 3
1 1 4
3 3 4

Output
YES
YES
NO
NO
```

Walkthrough: add `1-2`; query `1-2` -> `YES`. Add `2-3`; query `1-3` -> `YES` (path `1-2-3`).
Remove `1-2`; query `1-3` -> `NO` (vertex `1` is now isolated from `2-3`). Add `1-4`; query `3-4`
-> `NO` (`{1,4}` and `{2,3}` are separate components).

## Background

With **insertions only**, a disjoint-set union (DSU / union-find) answers everything online in near
`O(alpha)` amortized: each edge merges two components and a query is two `find`s. The trouble is the
removal. A DSU has no `delete` operation — once two trees are merged, the merge cannot be unwound in
better than rebuild time, because path compression has rewired arbitrarily many parent pointers. So
the central tension is: components are trivial to *grow* and effectively impossible to *shrink*.

Two families of approach are on the table before committing to one:

- **Recompute per query.** Maintain the live edge set in a hash structure; on each query run a BFS/DFS
  over the present edges. This is obviously correct and easy to write, but each query costs
  `O(n + m)`, so the worst case is `O(q*(n+m))` — quadratic, far outside the limits.
- **Make the whole problem offline.** All `q` operations are known up front, so each edge occupies a
  *contiguous interval of time* `[t_add, t_remove)` on the operation timeline. If we could "turn an
  edge on" for exactly the queries inside its interval and "turn it off" everywhere else, every query
  would see precisely its live edge set. That reframes a delete-heavy online problem as an interval
  cover, which is the shape a divide-and-conquer over time can exploit — provided the DSU can be made
  reversible.

## Evaluation settings

Judged on hidden tests covering: graphs that only ever grow (insert-only), heavy churn where edges
are repeatedly added and removed, edges that are added and never removed (alive to the end of the
timeline), queries that occur before any edge exists, self-queries (`u == v`), long path graphs that
stress union-find tree depth, `q = 0`, and the maximum scale `n = q = 2*10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    for (int t = 0; t < q; t++) {
        int type, u, v;
        cin >> type >> u >> v;
        // type 1: add edge u-v ; type 2: remove edge u-v ; type 3: query u-v.
        // TODO: answer each connectivity query under fully dynamic edges.
    }

    // TODO: output YES / NO per query, in order.
    return 0;
}
```

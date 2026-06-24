# Counting edges that lie on at least one cycle (loop audit)

## Research question

A data-center operator models its physical cabling as an **undirected multigraph**: `n` switches and `m`
cables, where a cable is an unordered pair of switch endpoints. The cabling may contain **parallel
cables** (two or more cables between the same pair of switches) and **self-loops** (a cable whose two
ends plug into the same switch — a slack loop left in the rack). The operator wants to audit
*redundancy*: a cable is **redundant** if its two endpoints would still be connected after that one
cable is unplugged, i.e. the cable lies on **at least one cycle**. A cable that is **not** redundant is
a **bridge**: unplugging it strictly splits a connected group of switches.

Output the number of redundant cables — equivalently, `m` minus the number of bridges. This is a
direct application of DFS bridge-finding, but the multigraph features (parallel edges, self-loops) are
exactly where a counting/dedup version of the algorithm goes subtly wrong.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m` (`0 <= n <= 2*10^5`, `0 <= m <= 2*10^5`).
  Then `m` lines follow, each with two integers `a b` (`1 <= a, b <= n`) describing one cable between
  switch `a` and switch `b`. `a == b` is allowed (a self-loop). Repeated pairs are allowed (parallel
  cables). Each cable is a distinct object even if another cable connects the same pair.
- Output (stdout): a single line with the number of cables that lie on at least one cycle.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the graph with `n = 6` and cables
`(1,2), (2,3), (3,1), (3,4), (4,5), (4,5), (5,6)`
the answer is `5`. The triangle `1-2-3` contributes its 3 cables; the two **parallel** cables `4-5`
each lie on the 2-cycle they form together, contributing 2 more; the cables `3-4` and `5-6` are
bridges and are excluded. `3 + 2 = 5`.

## Background

A **bridge** of an undirected graph is an edge whose removal increases the number of connected
components; equivalently, an edge that lies on **no** cycle. The standard linear-time detector is a
single DFS that, for each vertex `u`, computes `disc[u]` (the time `u` was first discovered) and
`low[u]` (the smallest `disc` reachable from `u`'s DFS subtree using at most one back edge). A tree
edge `u -> v` is a bridge **iff** `low[v] > disc[u]`: the subtree below `v` has no back edge climbing
to `u` or above, so nothing reconnects `v`'s side once that edge is cut.

The whole subtlety in a *multigraph* is the handling of the edge back to the parent. The textbook
trick "ignore the edge to the parent vertex" is wrong here: if there are **two** cables between a
vertex and its parent, the second one is a genuine back edge that makes both cables non-bridges, and a
vertex-based skip silently discards it. The correct discipline is to skip the **specific parent edge
instance** (by edge id), not every edge to the parent vertex. Self-loops are a second corner: a
self-loop can never be a bridge (its endpoints are trivially still connected), so it must count toward
the answer without ever being misclassified.

Two approaches are on the table before committing:

- **Per-edge connectivity test.** For each of the `m` cables, delete it and test whether its endpoints
  are still connected with a fresh BFS/DFS over the remaining `m-1` cables. This is obviously correct
  but `O(m * (n + m))` — fine for a brute-force oracle on tiny graphs, far too slow for `m = 2*10^5`.
- **Single-pass DFS bridge-finding.** One DFS computing `disc`/`low`, marking bridges by the
  `low[v] > disc[u]` test, then answering `m - (#bridges)`. This is `O(n + m)`; the open questions are
  the exact `low` recurrence, the parent-edge-instance handling, and self-loops.

## Evaluation settings

Judged on hidden tests covering: trees (every cable a bridge, answer `0`), single big cycle (answer
`m`), graphs with **parallel cables** between the same pair (both must count), **self-loops** (must
count), disconnected graphs (multiple DFS roots), the empty graph (`m = 0`, answer `0`), graphs with
isolated vertices, and large adversarial inputs (`n, m = 2*10^5`, including a single path of `2*10^5`
vertices that stresses recursion depth, and a star of parallel cables).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<array<int,2>> adj_[200005]; // {neighbor, edgeId}
int disc[200005], low_[200005], timer_;
bool isBridge[400005];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m)) return 0;
    for (int i = 0; i < m; i++) {
        int a, b; cin >> a >> b;
        adj_[a].push_back({b, i});
        if (a != b) adj_[b].push_back({a, i}); // self-loop stored once
    }

    // TODO: run DFS bridge-finding, mark each bridge edge id, then output
    //       m - (number of bridges) = number of cables on at least one cycle.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

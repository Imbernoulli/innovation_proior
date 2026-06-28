# Distinct colors in every rooted subtree

## Research question

You are given a rooted tree on `n` nodes (the root is node `1`). Each node `i` carries a color
`c[i]`. For **every** node `v`, report the number of **distinct** colors that appear among the nodes
of `v`'s subtree (the subtree includes `v` itself).

This is the canonical "subtree statistic" query: a value defined on every rooted subtree that is
*not* a simple additive aggregate (you cannot just add children's answers, because the same color may
recur across children and must be counted once). It is the kind of subproblem that appears inside
offline tree queries, auto-complete / trie analytics, and competitive-programming "for each subtree
report ..." tasks, so computing all `n` answers within the time limit — not just one — is the point.

## Input / output contract

- Input (stdin):
  - line 1: integer `n` (`1 <= n <= 2*10^5`);
  - line 2: `n` integers `c[1..n]` (`1 <= c[i] <= 10^9`), the colors, whitespace-separated;
  - the next `n-1` lines: two integers `u v` (`1 <= u, v <= n`, `u != v`) each describing an
    undirected tree edge. The tree is rooted at node `1`; edge endpoints may be given in any order.
- Output (stdout): `n` lines. Line `i` is the number of distinct colors in the subtree of node `i`
  (`1`-indexed, in node order `1..n`).
- Time limit: 2 seconds. Memory: 256 MB.

Example. `n = 5`, colors `c = [1, 2, 1, 3, 2]`, edges `1-2, 1-3, 3-4, 3-5`.
Subtree of `4` is `{4}` -> color `{3}` -> `1`. Subtree of `5` -> `{2}` -> `1`. Subtree of `2` ->
`{2}` -> `1`. Subtree of `3` is `{3,4,5}` -> colors `{1,3,2}` -> `3`. Subtree of `1` is all nodes ->
colors `{1,2,3}` -> `3`. Output: `3 / 1 / 3 / 1 / 1` (one per line).

## Background

The answer for `v` is `|{ c[w] : w in subtree(v) }|`. Two routes are on the table before committing:

- **Recompute per subtree.** For each `v`, walk its subtree and drop colors into a hash set, then read
  off the set size. Correct and trivial, but a node near the root is walked once for itself and once
  inside every ancestor's walk, so on a path of `n` nodes this is `Theta(n^2)` total work — hopeless
  at `n = 2*10^5`.
- **Merge children's structures bottom-up.** Keep a per-node container of the colors in its subtree and
  combine children into the parent. The open question is *how* to combine without paying for the heavy
  child every time: a generic "merge the two sets" is `O(n log^2 n)`; the order in which children are
  folded in is what decides whether the total is near-linear or quadratic.

## Evaluation settings

Judged on hidden tests covering: a single node (`n = 1`); all nodes the same color (every answer `1`);
all colors distinct (a leaf answers `1`, the root answers `n`); deep paths (`n = 2*10^5`, which stress
recursion depth); wide stars; random trees; and large/arbitrary color values up to `10^9` (so colors
must be coordinate-compressed, not used as array indices directly).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> color(n);
    for (auto &x : color) cin >> x;

    vector<vector<int>> g(n);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    // TODO: for each node v, output the number of distinct colors in v's subtree
    //       (tree rooted at node 0), one answer per line in node order.

    return 0;
}
```

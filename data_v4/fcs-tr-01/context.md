# Auxiliary (virtual) tree queries: isolating important cities

## Research question

A kingdom has `n` cities connected by `n - 1` bidirectional roads so that every city is reachable from
every other — i.e. the road network is a tree. We receive `q` independent queries. Each query names a set
`S` of **important** cities (the cities currently hosting ministries). For that query we must answer:

> What is the minimum number of **non-important** cities we have to capture (remove from the tree) so that,
> after their removal, no two important cities remain in the same connected component?

If two important cities of the query are directly joined by a road, there is no non-important vertex between
them to remove, so the task is impossible and the answer for that query is `-1`. Each query is answered
independently of the others (a captured city is "restored" before the next query).

The catch is scale: both the tree and the total size of the query sets are large, but each individual `S` is
typically tiny. The research question is how to answer every query in time proportional to `|S|` rather than
to `n`.

## Input / output contract

- Input (stdin):
  - line 1: integer `n` (`1 <= n <= 2*10^5`).
  - next `n - 1` lines: two integers `u v` (`1 <= u, v <= n`, `u != v`), a road between cities `u` and `v`.
    The edges form a tree. (For `n = 1` there are no edge lines.)
  - next line: integer `q` (`1 <= q <= 2*10^5`).
  - next `q` queries: each begins with `k` (`1 <= k <= n`), the number of important cities, followed by `k`
    distinct city indices. The sum of `k` over all queries satisfies `sum k <= 2*10^5`.
- Output (stdout): for each query, one line — the minimum number of non-important cities to capture, or `-1`
  if two important cities are adjacent.
- Time limit: 2 seconds. Memory: 256 MB.

Worked sample. Input:

```
7
1 2
1 3
2 4
2 5
3 6
3 7
4
2 4 5
2 6 7
3 4 6 7
2 1 2
```

Output:

```
1
1
1
-1
```

The tree is rooted at `1` with children `2` and `3`; city `2` has children `4, 5` and city `3` has children
`6, 7`. Query `{4, 5}`: capture city `2` to split `4` from `5` — one deletion. Query `{6, 7}`: capture `3` —
one deletion. Query `{4, 6, 7}`: capturing city `3` isolates `6` and `7` from each other, and `4` is already
in a different part of the tree — one deletion. Query `{1, 2}`: cities `1` and `2` are joined by a road and
both important, so it is impossible — `-1`.

## Background

This is the classic "make every important vertex its own island" tree problem. Two facts shape any solution.

First, a **structural** observation about *which* vertices can ever be worth deleting: an optimal solution only
ever deletes a vertex that is either the lowest common ancestor (LCA) of two important vertices, or lies on a
tree path between two important vertices. A vertex with no important vertex in two different subtrees below it
is never useful to delete. So the only geometry that matters for a query is the set `S` together with the
pairwise LCAs of `S`.

Second, given a rooted tree, there is a clean **bottom-up DP** that computes the answer in one pass:

- mark impossible if any important vertex has an important parent;
- otherwise, for each vertex `v` in post-order, let `cnt[v]` be the number of important vertices still
  "connected up" to `v` through its subtree. If `v` is important, every child branch that still carries a
  connected important vertex must be severed (one deletion per such branch) and `cnt[v] = 1`. If `v` is not
  important and `s = sum of children cnt` is `>= 2`, delete `v` itself (one deletion) and set `cnt[v] = 0`;
  if `s == 1`, pass it up (`cnt[v] = 1`); if `s == 0`, `cnt[v] = 0`.

Run on the **whole** tree this DP is `O(n)` per query, hence `O(q n)` overall — far too slow at `n, q ~ 2*10^5`.
The open problem is to run the *same* DP but only over the part of the tree that matters for `S`.

## Evaluation settings

Judged on hidden tests covering: single-vertex queries (always `0`); queries with two adjacent important
cities (`-1`); chains/paths, stars, caterpillars and broom-shaped trees; deep trees that stress recursion;
the maximum scale `n = 2*10^5`, `q = 10^5`, `sum k = 2*10^5`; and queries whose important set spans many
subtrees. Outputs are compared exactly, line by line.

## Code framework

A single self-contained C++17 program reading stdin and writing stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // TODO: root the tree; precompute LCA structure.

    int q; cin >> q;
    while (q--) {
        int k; cin >> k;
        vector<int> S(k);
        for (auto &x : S) cin >> x;

        // TODO: answer this query in O(|S| log n) — over S and its LCAs only,
        //       not the whole tree.
        long long answer = 0;
        cout << answer << "\n";
    }
    return 0;
}
```

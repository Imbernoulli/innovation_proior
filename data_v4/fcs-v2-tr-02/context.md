# Constant-time k-th ancestor with many queries on a static tree

## Research question

You are given a rooted tree on `n` nodes (labeled `1..n`) and must answer `q` "level ancestor"
queries. Each query gives a node `v` and a non-negative integer `k`, and asks for the **k-th
ancestor** of `v` — the node reached by climbing exactly `k` parent edges from `v` toward the root.
If `v` has fewer than `k` ancestors (i.e. `k` exceeds the depth of `v`), there is no such node.

Both `n` and `q` are large (up to `5*10^5`). The tree is fixed: all edges are known up front, and
no updates occur between queries. The challenge is to make **each query answer in O(1) time** after
an affordable preprocessing pass, rather than spending logarithmic or linear time per query.

This is the classic *level-ancestor problem* (LA). It is a building block inside LCA structures,
auto-completion / trie navigation, suffix-tree algorithms, and any setting where you repeatedly need
"the ancestor a fixed number of hops up" on an unchanging hierarchy.

## Input / output contract

- Input (stdin):
  - The first token is `n` (`0 <= n <= 5*10^5`), the number of nodes.
  - Then `n` integers `par[1], par[2], ..., par[n]`, where `par[i]` is the parent label of node `i`,
    or `0` if node `i` is the root. Exactly one node has parent `0`. The `par` array describes a
    valid rooted tree (no cycles); parent labels may be larger or smaller than the child label.
  - Then the integer `q` (`0 <= q <= 5*10^5`), the number of queries.
  - Then `q` queries, each a pair `v k` (`1 <= v <= n`, `0 <= k <= 10^9`).
- Output (stdout): for each query, one line with the label of the k-th ancestor of `v`, or `0` if no
  such ancestor exists. (When `k = 0` the answer is `v` itself.)
- Time limit: 1 second. Memory: 256 MB.

Example:

```
6
0 1 1 2 3 4
9
6 0
6 1
6 2
6 3
6 4
5 1
5 2
4 2
3 1
```

The tree is `1 -> {2,3}`, `2 -> 4`, `3 -> 5`, `4 -> 6` (depths: node 1 at 0, nodes 2,3 at 1, nodes
4,5 at 2, node 6 at 3). The expected output is:

```
6
4
2
1
0
3
1
1
1
```

(`6 0` is node 6 itself; `6 1=4`, `6 2=2`, `6 3=1`; `6 4=0` because node 6 has only 3 ancestors;
`5 1=3`, `5 2=1`; `4 2=1`; `3 1=1`.)

## Evaluation settings

Judged on hidden tests covering: trees of many shapes (random, caterpillar/path of depth up to `n`,
star, skewed/long-path); root labels that are not node `1` and parent labels that point "forward" to
larger labels; queries with `k = 0`, `k` exactly equal to the depth (the root), `k` one past the
depth, and very large `k` (up to `10^9`, far beyond any depth); the single-node tree (`n = 1`) and
the empty input (`n = 0`); and full-scale `n = q = 5*10^5` where an `O((n + q) log n)` query loop
would be at risk on a 1-second limit and an O(1)-query structure is comfortable.

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

    vector<int> par(n + 1, 0);
    for (int v = 1; v <= n; v++) cin >> par[v];

    // TODO: preprocess the static tree so that each k-th-ancestor query is answered in O(1).

    int q;
    cin >> q;
    for (int Q = 0; Q < q; Q++) {
        int v; long long k;
        cin >> v >> k;
        long long ans = 0; // TODO: k-th ancestor of v, or 0 if it does not exist
        cout << ans << "\n";
    }
    return 0;
}
```

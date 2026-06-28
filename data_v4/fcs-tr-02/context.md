# Counting tree paths whose edge-weight sum is exactly L

## Research question

You are given a tree with `n` vertices and `n - 1` weighted edges; every edge weight is a positive
integer. For a fixed target `L`, count the number of **unordered pairs of distinct vertices**
`(u, v)` such that the sum of the edge weights along the unique path between `u` and `v` equals
**exactly** `L`.

The unique-path property of a tree means each pair `(u, v)` defines one well-formed distance, so the
quantity is well defined. The challenge is purely scale: with `n` up to `2 * 10^5` there are about
`2 * 10^10` pairs, so anything that touches pairs individually is hopeless. This is the canonical
"path counting on a tree" problem that the centroid-decomposition technique was built for.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `L` (`1 <= n <= 2*10^5`, `0 <= L <= 10^6`).
  - The next `n - 1` lines each have three integers `u v w` (`1 <= u, v <= n`, `u != v`,
    `1 <= w <= 10^6`) describing an edge between `u` and `v` of weight `w`.
  - The edges are guaranteed to form a tree. When `n = 1` there are no edge lines.
- Output (stdout): a single line with the number of unordered pairs whose path weight is exactly `L`.
- Time limit: 2 seconds. Memory: 256 MB.

The answer can exceed a 32-bit integer (a star with `~2*10^5` equal-weight leaves and `L = 2w` yields
about `2 * 10^10` pairs), so it must be accumulated in a 64-bit type.

Example:

```
6 6
1 2 5
2 3 5
2 4 5
1 5 1
5 6 5
```

Vertex `6` sits at distance `1 + 5 = 6` from vertex `1` (path `6 - 5 - 1`), and vertex `5` sits at
distance `1 + 5 = 6` from vertex `2` (path `5 - 1 - 2`). No other pair sums to `6`, so the two
qualifying pairs are `(1, 6)` and `(2, 5)`, and the answer is `2`.

## Background

The constraint that the path weight be **exactly** `L` (not "at most", not "shortest") is what makes
this a counting problem rather than an optimization. Two families of approach are on the table before
committing:

- **All-pairs distance.** Root the tree, or BFS/DFS from every vertex, and for each ordered or
  unordered pair read off the path weight, incrementing a counter when it equals `L`. This is
  `O(n^2)` and obviously correct, which makes it the natural oracle, but at `n = 2*10^5` it is far
  out of reach.
- **Divide and conquer on the tree via centroids.** Every path either passes through a chosen
  "centroid" vertex or lies entirely inside one of the components left when the centroid is removed.
  Counting the paths through the centroid is a one-dimensional frequency problem; the rest is solved
  by recursion on each component. Choosing the centroid (a vertex whose removal leaves every piece
  with at most half the vertices) bounds the recursion depth to `O(log n)`.

## Evaluation settings

Judged on hidden tests covering: `n = 1` (no pairs, answer `0`); `n = 2` (single edge, answer `1`
iff its weight equals `L`); long path/chain trees; star trees (which stress the 64-bit answer);
random bushy trees; `L = 0` (unreachable with positive weights, answer `0`); `L` exactly equal to the
tree diameter (one pair); and full-scale `n = 2*10^5` for the time limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;

    vector<vector<pair<int,int>>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // TODO: count unordered pairs (u, v) whose tree-path weight sum equals exactly L.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

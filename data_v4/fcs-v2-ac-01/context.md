# Minimum spanning tree using exactly k white edges

## Research question

You are given a connected, undirected, weighted graph on `n` vertices and `m` edges. Every edge is
coloured **white** (`c = 1`) or **black** (`c = 0`). Among all spanning trees of the graph, find the
one of **minimum total weight that uses exactly `k` white edges** (and therefore exactly `n - 1 - k`
black edges). Output that minimum total weight, or `-1` if no spanning tree with exactly `k` white
edges exists.

This is a *constrained* minimum-spanning-tree problem: the colour count is a hard equality constraint
layered on top of an ordinary MST. The constrained-resource-count pattern ("optimize a cost, but use
*exactly* `k` of some resource") appears in budgeted network design, in "use exactly `k` of the
premium links" provisioning problems, and as a subroutine inside larger optimizers. Getting the
exact-`k` corner right — including infeasible `k`, the boundary counts, and ties — is the whole task.

## Input / output contract

- Input (stdin):
  - The first line contains three integers `n m k`
    (`1 <= n <= 2*10^5`, `0 <= m <= 4*10^5`, `0 <= k <= n - 1`; if `k` is outside `[0, n-1]` the
    answer is `-1`).
  - Each of the next `m` lines contains four integers `u v w c`:
    an edge between vertices `u` and `v` (`1 <= u, v <= n`, `u != v`) of weight `w`
    (`0 <= w <= 10^9`) and colour `c` (`c = 1` white, `c = 0` black). Parallel edges and any edge
    distribution are allowed.
- Output (stdout): a single line with the minimum total weight of a spanning tree using exactly `k`
  white edges, or `-1` if none exists (including when the graph is disconnected).
- Time limit: 2 seconds. Memory: 256 MB.

Example: the graph

```
4 5 1
1 2 1 1
2 3 2 0
3 4 3 0
1 4 4 1
1 3 5 0
```

has answer `6`: take the white edge `1-2` (weight 1) and the two black edges `2-3` (weight 2) and
`3-4` (weight 3); that spanning tree uses exactly one white edge and totals `6`, and no exactly-one-
white spanning tree is cheaper.

## Background

The constraint "exactly `k` of one colour" is what makes this more than a plain MST. Two families of
approach are on the table before committing.

- **Per-count dynamic programming.** Track, while building the tree, how many white edges have been
  used, computing `dp[count]` = best weight achievable with that white count. A spanning-tree DP that
  carries the white count along is the obvious "make the constraint a state" move. The open question is
  whether such a DP can be both correct and fast enough at `n, m` up to a few hundred thousand.
- **Penalty / Lagrangian relaxation.** Add a per-white-edge penalty `lambda` to the objective, build an
  ordinary (unconstrained) MST under the penalized weights, and tune `lambda` so that the resulting
  tree happens to use exactly `k` white edges. The open questions are *why* a single scalar `lambda`
  suffices, *which* `lambda` to pick, and how to recover the true weight from the penalized one.

The function `f(k)` = (min weight of a spanning tree with exactly `k` white edges) is defined on a
contiguous interval `[minWhite, maxWhite]` of achievable white counts; outside it the answer is `-1`.
Whether `f` has structure that a penalty method can exploit is the crux of the problem.

## Evaluation settings

Judged on hidden tests covering: feasible `k` in the interior of the white-count window; the boundary
counts `k = minWhite` and `k = maxWhite`; infeasible `k` (too few or too many white edges achievable,
and `k > n - 1`); disconnected graphs (`-1`); all-white and all-black graphs; `n = 1` (the empty tree,
answer `0` only for `k = 0`); parallel edges; and large connected graphs with `n = 2*10^5`,
`m = 4*10^5`, weights near `10^9` so the total weight exceeds a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, k;
    if (!(cin >> n >> m >> k)) return 0;
    for (int i = 0; i < m; i++) {
        int u, v, w, c;
        cin >> u >> v >> w >> c;
        // u, v in [1, n]; w in [0, 1e9]; c in {0 (black), 1 (white)}.
        // TODO: store the edge.
    }

    // TODO: compute the minimum-weight spanning tree using exactly k white edges,
    // or -1 if no such spanning tree exists.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```

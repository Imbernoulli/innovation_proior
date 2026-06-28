# Relay coverage clusters: connected subsets through each station

## Research question

A communication grid is laid out as a **tree**: there are `n` relay stations (numbered `1..n`)
and exactly `n-1` bidirectional fibre links, and the whole grid is connected, so between any two
stations there is exactly one path. A maintenance crew can power up any set `S` of stations as a
single **coverage cluster**, but a cluster is only valid if the powered stations form a *connected*
region of the grid: the subgraph induced by `S` must be connected (every two stations in `S` are
joined by a path that stays inside `S`).

For each station `r` the operator wants to know how many distinct **valid clusters contain `r`** —
that is, the number of connected vertex subsets `S` of the tree with `r in S`. The empty set is not
a cluster (a cluster that contains `r` has at least the one station `r`, which by itself is
connected and counts). Because these counts grow astronomically, report each one **modulo
`1000000007`**.

So the deliverable is a vector of `n` numbers: the `r`-th is the number of connected subsets of the
tree containing vertex `r`, taken mod `1000000007`.

This is the rooted-subtree-counting question in its purest form. The same "count connected pieces
seen from every vertex" primitive sits underneath influence/spread models, articulation analysis,
and tree-DP feature extraction, so the per-vertex (not just per-tree) version is the one worth
getting exactly right.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`). Then follow `n-1` lines, each with two
  integers `u v` (`1 <= u, v <= n`, `u != v`) describing a fibre link between stations `u` and `v`.
  The links are guaranteed to form a tree. When `n = 1` there are no link lines.
- Output (stdout): a single line with `n` integers separated by single spaces. The `r`-th integer is
  the number of connected subsets containing station `r`, modulo `1000000007`. Terminate the line
  with a newline.
- Time limit: 1 second. Memory: 256 MB.

Example: for the path `1 - 2 - 3 - 4 - 5` the answer is `5 8 9 8 5`. A connected subset of a path is
a contiguous segment; the segments containing vertex `1` are `[1..k]` for `k = 1..5`, giving `5`,
and the segments containing the middle vertex `3` are `[i..j]` with `i in {1,2,3}`, `j in {3,4,5}`,
giving `3 * 3 = 9`.

## Background

The count for a single vertex is a textbook rooted-tree DP. Root the tree at `r`; then a connected
subset containing `r` is built by, at every vertex `v` already in the subset, independently choosing
for each child `c` whether to *not* extend into `c`'s subtree, or to extend by including `c` and then
recursing. That gives the multiplicative recurrence `down[v] = prod over children c of (1 + down[c])`,
and the answer for root `r` is `down[r]`. Computing this for one fixed `r` is a single `O(n)`
post-order pass.

Two families of approach are on the table before committing to one:

- **Recompute per root.** Loop over all `n` choices of `r`, and for each run the `O(n)` rooted DP from
  scratch. It is `O(n^2)`, dead simple, and obviously correct; the open question is whether it can
  possibly fit inside the time limit at `n = 2*10^5`.
- **Reroot in two passes.** Compute the DP once for a fixed root, then in a second pass *transport*
  the root from a vertex to each of its neighbours using only local information, so every vertex's
  answer is produced in `O(1)` amortised. It is `O(n)`; the open question is the exact rule for
  combining a vertex's "downward" subtrees with the single "upward" direction without recomputation,
  and how to do that combination safely under the modulus.

## Evaluation settings

Judged on hidden tests covering: `n = 1` (answer `1`); `n = 2`; long paths (where contiguous-segment
counts are easy to verify by hand); stars (a centre with many leaves); deep skewed/caterpillar trees
(to stress an iterative, non-recursive traversal); broad random trees; and maximal `n = 2*10^5` with
arbitrary shape, where the counts wrap around the modulus many times. Edges may be listed in any
order and either orientation; the vertex `1` is not assumed to be a leaf, the centre, or special in
any way.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // TODO: for every vertex r, compute the number of connected subsets of the
    // tree that contain r, modulo MOD; output the n results on one line.

    return 0;
}
```

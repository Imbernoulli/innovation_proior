# Forging a server network: counting connected pairs as cables are installed

## Research question

A data center has `n` servers numbered `1..n`. A crew installs `m` cables one at a time. Each cable
directly connects two servers `u` and `v`; once two servers are joined by a path of cables they are
in the same *cluster* and can talk to each other.

After installing each cable, report how many **unordered pairs of distinct servers are currently in
the same cluster** (i.e. can reach each other). A cable may be redundant — if `u` and `v` are already
in the same cluster (including the degenerate case `u == v`), it connects nothing new and the count
is unchanged. After all `m` cables, also report the **grand total**: the sum of the `m` per-cable
counts you printed.

Concretely, if the clusters currently have sizes `c_1, c_2, ...`, the connected-pair count is
`sum_k C(c_k, 2) = sum_k c_k*(c_k-1)/2`. You must output this value after every cable, and finally
the running sum of all of them.

This is the classic incremental-connectivity setting (union–find / DSU), but the quantity being
tracked grows quadratically in cluster size and is then summed over time, so the numbers get large
fast — the interesting part is keeping the arithmetic exact.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`). Then `m` lines follow, each with two integers `u` and `v`
  (`1 <= u, v <= n`) describing a cable in installation order. `u == v` is allowed.
- Output (stdout): `m + 1` lines. The first `m` lines: after installing cable `i`, the number of
  unordered pairs of distinct servers in the same cluster. The final line: the grand total, i.e. the
  sum of those `m` printed values. (If `m == 0`, print only the grand total `0`.)
- Time limit: 1 second. Memory: 256 MB.

Example: for `5` servers and the cables `(1,2), (3,4), (2,3), (3,4), (1,5)` the per-cable counts are
`1, 2, 6, 6, 10` and the grand total is `25`.

## Background

Two facts drive the design:

- **Incremental connectivity wants union–find.** Rebuilding components from scratch after each cable
  is `O(m * n)` and far too slow at `n, m = 2*10^5`. A disjoint-set-union (DSU) structure with union
  by size and path compression maintains the cluster of each server in near-constant amortized time,
  and crucially lets us know the size of each cluster being merged.

- **The count updates by a single product.** Let the running count be `P`. Merging two *distinct*
  clusters of sizes `sA` and `sB` creates exactly `sA * sB` new same-cluster pairs (every server of
  one cluster pairs with every server of the other), and changes nothing else, so `P += sA * sB`. A
  redundant cable leaves `P` unchanged. This avoids recomputing `sum C(c_k, 2)` from scratch.

The open arithmetic question — and the reason this problem exists — is the magnitude of the numbers.
With `n = 2*10^5`, a single fully merged cluster contributes `C(2*10^5, 2) ≈ 2*10^10` pairs, and even
one merge step `sA * sB` can be `~10^10`. Summed over up to `2*10^5` cables, the grand total reaches
`~10^15`. None of these fit in a signed 32-bit integer (max `~2.1*10^9`).

## Evaluation settings

Judged on hidden tests covering: `m = 0`; `n = 1`; redundant cables and self-loops (`u == v`); a
long chain that grows one giant cluster (so the per-cable count climbs to `C(n, 2)`); many redundant
cables after full connection (so the grand total balloons); and worst-case `n = m = 2*10^5` random
cables. Several tests are tuned so that any accumulator or any intermediate product computed in
32-bit `int` silently overflows and produces a wrong answer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int parent_[200005];
long long sz[200005];

int find_(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]]; // path halving
        x = parent_[x];
    }
    return x;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    for (int i = 1; i <= n; i++) { parent_[i] = i; sz[i] = 1; }

    // TODO: process each cable with DSU, maintain the connected-pair count P,
    //       print P after each cable, and accumulate the grand total.

    return 0;
}
```

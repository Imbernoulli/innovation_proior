**Problem.** `n` servers, `m` cables installed one at a time; each cable joins servers `u` and `v` (self-loops `u == v` allowed). Two servers are in the same cluster if connected by a path of cables. After each cable, print the number of unordered pairs of distinct servers currently in the same cluster, i.e. `sum_k C(c_k, 2)` over cluster sizes `c_k`. After all cables, print the grand total: the sum of those `m` printed values. Read from stdin, write `m + 1` lines to stdout.

**Key idea — DSU with an incremental product update.** Maintain disjoint-set-union (union by size, path halving) so each find is near-`O(α(n))`. Keep a single running count `P`. When a cable joins two *distinct* clusters of sizes `sA` and `sB`, every server of one becomes connected to every server of the other and nothing else changes, so `P += sA * sB`. A cable whose endpoints already share a cluster (including any self-loop) is redundant and leaves `P` unchanged — a single `if (root(u) != root(v))` guard handles both. Print `P` after each cable; accumulate `P` into the grand total each step; print the grand total at the end.

**Correctness.** `P` equals `sum_k C(c_k, 2)` at all times. Base: all singletons give `0 = sum C(1,2)`. Step: merging sizes `sA, sB` replaces terms `C(sA,2) + C(sB,2)` with `C(sA+sB,2)`, and `C(sA+sB,2) - C(sA,2) - C(sB,2) = sA*sB`, exactly the increment applied; a redundant cable changes no cluster so `P` is unchanged. Hence `P` is always the true connected-pair count, and the grand total is the sum of these. The monotonic increase of `P` (a connected-pair count can never drop as edges are added) is a built-in sanity invariant.

**Pitfalls.**
1. *Int overflow in the product (the deadly one).* `sz[ru] * sz[rv]` can reach `~(10^5)^2 = 10^10`. If `sz` is an `int` array, the multiplication is done in 32-bit and overflows *before* it is widened into a `long long` accumulator — so declaring `long long pairs` is **not** enough. Concretely, merging two size-`50000` clusters computes `50000*50000 = 2.5*10^9` which wraps to a negative `int` and makes the count *decrease*. Fix: store sizes as `long long` (or cast: `(long long)sz[ru] * sz[rv]`).
2. *Accumulator magnitude.* Peak per-cable count is `C(2*10^5, 2) = 19,999,900,000 ≈ 2*10^10`; the grand total can reach `~4*10^15`. Both need `long long`; an `int` is a silent wrong answer on the large tests.
3. *Read sizes before the union.* Compute `sz[ru]*sz[rv]` *before* doing `sz[ru] += sz[rv]`, else you multiply post-merge sizes.
4. *Don't add for redundant cables.* Without the `root(u) != root(v)` guard, an edge inside an existing cluster would wrongly add `sz*sz` and over-count.

**Edge cases.** `m = 0`: no per-cable lines, print grand total `0`. `n = 1`: no distinct pairs ever; every cable is a self-loop, all lines `0`. Self-loops and redundant edges: absorbed by the guard. Worst case `n = m = 2*10^5`: ~70 ms, ~9 MB — well within limits.

**Complexity.** `O((n + m) α(n))` time (effectively linear), `O(n)` space. Output is built in one string and flushed once to avoid per-line stream overhead.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int parent_[200005];
long long sz[200005];

int find_(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
    }
    return x;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 1; i <= n; i++) {
        parent_[i] = i;
        sz[i] = 1;
    }

    long long pairs = 0;     // current number of connected unordered pairs
    long long grand = 0;     // running sum of "pairs" reported after each cable

    string out;
    out.reserve((size_t)m * 8);

    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        int ru = find_(u), rv = find_(v);
        if (ru != rv) {
            // merging two distinct components adds sz[ru]*sz[rv] new connected pairs
            pairs += sz[ru] * sz[rv];
            // union by size
            if (sz[ru] < sz[rv]) swap(ru, rv);
            parent_[rv] = ru;
            sz[ru] += sz[rv];
        }
        // if ru == rv, the cable is redundant; pairs unchanged
        grand += pairs;
        out += to_string(pairs);
        out += '\n';
    }

    out += to_string(grand);
    out += '\n';
    cout << out;
    return 0;
}
```

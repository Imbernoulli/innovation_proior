# Counting redundant cables and same-circuit pairs as a network grows

## Research question

A data center has `n` routers numbered `1..n` and no cables yet. Technicians install `m` cables one at a
time; cable `e` connects two **distinct** routers `u_e` and `v_e`. A cable is called **redundant** if, at the
moment it is installed, its two endpoints are already connected through previously installed cables (so the
new cable closes a cycle and does not merge two separate connected groups). Duplicate cables between the same
pair of routers are allowed and every cable after the first such pair is redundant.

After all `m` cables are installed, report three integers:

1. `R` — the total number of redundant cables.
2. `P` — the number of **unordered pairs** of distinct routers `{a, b}` that end up in the same connected
   component (i.e. `sum over components C(size, 2)`).
3. `S` — the running total `sum_{e=1..m} R_e`, where `R_e` is the number of redundant cables among the first
   `e` cables. (Equivalently: after installing each cable, look at how many redundant cables you have counted
   so far, and add that number to `S`.)

`P` is a same-component pair count; `R` and `S` are cycle-closing counts. All three are easy to get *subtly*
wrong by double-counting cross pairs, mis-deriving `C(size, 2)`, or letting a 32-bit accumulator overflow.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m` (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`). Then
  `m` lines follow, each with two integers `u_e v_e` (`1 <= u_e, v_e <= n`, `u_e != v_e`).
- Output (stdout): a single line with three space-separated integers `R P S`.
- Time limit: 1 second. Memory: 256 MB.

Example: with `n = 5` and cables `(1,2), (2,3), (1,3), (4,5), (4,5)` the answer is `2 4 4`.

## Background

The natural data structure is a **disjoint-set union** (union–find) over the routers. Each cable triggers one
query: are `u_e` and `v_e` already in the same set?

- If `find(u_e) == find(v_e)`, the endpoints are already connected, so the cable is redundant — it must
  increment `R` and contribute **zero** new same-component pairs.
- Otherwise it merges two components of sizes `s1` and `s2`. The number of *new* unordered same-component
  pairs created is exactly `s1 * s2`: every router in one component pairs with every router in the other, and
  each such cross pair is counted once. Pairs that were already inside a single component are untouched.

Two families of approach are on the table before committing:

- **Recompute from scratch after each cable.** Rebuild components with BFS/DFS and re-sum `C(size, 2)`. This
  is obviously correct but `O(n * m)` in the worst case — far too slow for the stated bounds. Useful only as
  an independent oracle on tiny inputs.
- **Incremental DSU with running counters.** Maintain `samePairs` and add `s1 * s2` only on a real merge,
  `R` only on a redundant cable, and fold `R` into `S` after every cable. This is `O((n + m) * alpha(n))`. The
  open question is the exact increment (`s1 * s2` vs. an ordered or doubled variant) and the integer width.

## Evaluation settings

Judged on hidden tests covering: `m = 0` (no cables, answer `0 0 0`); a single router (`n = 1`, forcing
`m = 0`); long chains where every cable merges (so `R = 0` but `P` reaches `C(n, 2) ~ 2*10^10`); floods of
duplicate cables (so `R` and therefore `S` reach `~2*10^10`); and mixed graphs that exercise the
distinction between redundant cables and genuine merges. Several tests are chosen so that 32-bit accumulators
overflow on `P` and `S`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int par[200005], sz[200005];

int find(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 1; i <= n; i++) { par[i] = i; sz[i] = 1; }

    long long redundant = 0, samePairs = 0, prefixRedundantSum = 0;

    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        // TODO: union-find query; update redundant / samePairs / prefixRedundantSum correctly.
    }

    cout << redundant << " " << samePairs << " " << prefixRedundantSum << "\n";
    return 0;
}
```

**Problem.** There are `n` stars in a line, numbered `1..n`. You apply `m` operations; operation `(l, r)` (`1 <= l <= r <= n`) makes every star in the inclusive range `l, l+1, ..., r` mutually connected. Bridges are permanent and operations may overlap, repeat, or be single points. After all operations, output the number of connected clusters (a star never linked to a neighbour is its own singleton). Read `n`, `m`, then the `m` pairs from stdin; print one integer.

**Key idea — skip-pointer DSU for range union.** A range `[l, r]` is realised by the chain of `r - l` adjacency links with left endpoints `l, l+1, ..., r - 1` (link `i` joins stars `i` and `i+1`). Linking every pair in every range is correct but `O(sum of range lengths)` = quadratic on many long overlapping ranges. Instead keep a second pointer `nxt[i]` = "smallest index `>= i` whose outgoing link has not yet been built". To process `(l, r)`: start at `i = fnext(l)`, and while `i < r`, build `unite(i, i+1)`, consume `i` via `nxt[i] = i + 1`, and advance `i = fnext(i + 1)`. Each link is built at most once over the whole run, so total `unite` calls `<= n - 1`. Finally count DSU roots among `1..n`.

**Correctness.** The walk builds exactly the links with left endpoints `l .. r-1` that are still unbuilt; already-built links are skipped by `nxt[]` (which only ever points forward to the first active index), so the connectivity result is identical to linking every pair, just without the redundant work. Counting roots after all unions gives the cluster count by definition of DSU.

**Pitfalls.**
1. *Inclusive range vs exclusive loop (the off-by-one).* The range is inclusive in `r` for *stars*, but the loop over *left-endpoint links* must stop **before** `r`. Use `while (i < r)`, not `while (i <= r)`. With `i <= r`, when the cursor reaches `i = r` you call `unite(r, r+1)` — a link reaching **outside** the range that wrongly glues star `r` to star `r+1`. A trace of the two-star range `(1,2)` on `n = 3` exposes it: the buggy guard returns `1` cluster instead of the correct `2`. The strict guard also makes single-point ranges `(l, l)` build zero links automatically (`l < l` is false).
2. *Quadratic time without the skip pointer.* Re-linking already-connected pairs (e.g. `(1,n)` applied `m` times) is `O(n*m)` and times out; the `nxt[]` skip pointer makes each link unique, giving near-linear total work.

**Edge cases.** `m = 0` -> answer `n` (no links). `n = 1` -> answer `1` (no valid links exist). Fully overlapping/repeated ranges -> the skip pointer collapses redundant passes to `O(alpha)` each. Single-point range `(l, l)` -> zero links, cluster count unchanged. No arithmetic overflow: all indices `<= n + 1`, well within `int`; arrays are sized `n + 2` so `nxt[i+1]`/`par[i+1]` are always in bounds.

**Complexity.** `O((n + m) * alpha(n))` time (each adjacency link built once, path compression on both `par[]` and `nxt[]`), `O(n)` memory. The `n = m = 2*10^5` worst case runs in a few hundredths of a second.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// DSU over stars 1..n, with a "next free" skip pointer to union a whole
// contiguous range in near-linear amortized total time.
struct DSU {
    vector<int> par;     // connectivity parent
    vector<int> nxt;     // nxt[i] = smallest index >= i not yet "consumed" by a range-union step
    DSU(int n) : par(n + 2), nxt(n + 2) {
        for (int i = 0; i <= n + 1; i++) { par[i] = i; nxt[i] = i; }
    }
    int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a != b) par[a] = b;
    }
    // skip-pointer find over nxt[]
    int fnext(int x) { while (nxt[x] != x) { nxt[x] = nxt[nxt[x]]; x = nxt[x]; } return x; }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    DSU d(n);

    for (int op = 0; op < m; op++) {
        int l, r;
        cin >> l >> r;
        // Union all stars in [l, r] together. We walk from l upward using the
        // skip pointer nxt[] so each star is "consumed" (its nxt advanced) at
        // most once across the whole run, giving near-linear total work.
        //
        // Boundary: a range [l, r] has (r - l) adjacent links l-l+1, ..., (r-1)-r.
        // We start at the first not-yet-consumed index >= l, and keep linking it
        // to its successor while we are still strictly below r. The cursor i must
        // never pass r, and the LAST link we add is (r-1) -> r, so the loop runs
        // while i < r (NOT i <= r): linking i to i+1 when i == r would touch r+1,
        // which is outside the range.
        int i = d.fnext(l);
        while (i < r) {
            d.unite(i, i + 1);   // connect star i with star i+1
            d.nxt[i] = i + 1;    // i is consumed; future range-walks skip past it
            i = d.fnext(i + 1);  // jump to next not-yet-consumed index
        }
    }

    // Count connected components among all stars 1..n.
    int comps = 0;
    for (int v = 1; v <= n; v++)
        if (d.find(v) == v) comps++;

    cout << comps << "\n";
    return 0;
}
```

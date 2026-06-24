**Problem.** A network of `n` routers starts with no cables. `m` cables are installed one at a time, each joining two distinct routers `u, v`. A cable is *redundant* if its endpoints are already connected when it is installed. After all cables, output three integers: `R` = number of redundant cables; `P` = number of unordered pairs of distinct routers in the same final connected component (`sum over components C(size, 2)`); and `S = sum_{e=1..m} R_e`, where `R_e` is the count of redundant cables among the first `e` cables. Read input from stdin, print `R P S`.

**Key idea — incremental DSU with running counters.** Maintain a disjoint-set-union over routers. For each cable joining `u, v`, let `ru = find(u)`, `rv = find(v)`:

- If `ru == rv`, the cable is redundant: `R += 1`, and it creates **no** new same-component pair.
- Otherwise it merges two components of sizes `s1 = sz[ru]` and `s2 = sz[rv]`. The new unordered same-component pairs are exactly the cross pairs, `s1 * s2` of them, so `P += s1 * s2`. Then union.

After **every** cable (redundant or not), fold the current `R` into `S`: `S += R`. This realizes `S = sum_e R_e`, since after cable `e` the variable `R` equals `R_e`.

**Correctness.** A merge of components `ru, rv` makes same-component exactly the pairs `{a, b}` with `a` in `ru` and `b` in `rv`; there are `s1 * s2` such unordered pairs, each produced once because the two sides are disjoint. Pairs already inside a component are unaffected. Summing these increments over all real merges yields `P = sum over final components C(size, 2)` (telescoping: a component of size `k` is built by merges whose `s1*s2` increments sum to `C(k,2)`). Redundant cables add nothing to `P`, increment `R`, and do not change component structure. The per-cable `S += R` directly sums the running redundant count.

**Pitfalls.**
1. *Unordered vs. ordered (double-count).* The increment is `s1 * s2`, **not** `s1*s2 + s2*s1 = 2*s1*s2`. The doubled form counts each unordered pair twice. (Trace `n=3, (1,2),(2,3)`: the wrong version gives `P = 6`, the truth is `C(3,2) = 3`.)
2. *Running sum vs. increment (off-by-one in `S`).* `S` must accumulate the current `R` after *every* cable, not only when a redundant cable appears. Updating `S` only inside the redundant branch drops the contributions of later merges and undercounts. (Trace `n=3, (1,2),(1,2),(2,3)`: gives `S = 1` instead of `2`.)
3. *Read sizes before the union.* Capture `s1, s2` before `sz[ru] += sz[rv]`, or the product uses the merged size.
4. *Overflow.* `P` and `S` reach `~2*10^10` (e.g. `C(2*10^5, 2)` and `1+2+...+2*10^5`), and `s1*s2` reaches `~4*10^10` — all beyond 32-bit. Use `long long` for every counter and the product. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `m = 0` -> `0 0 0`. `n = 1` (forces `m = 0` since `u != v`) -> `0 0 0`. A full merging chain -> `R = 0`, `S = 0`, `P = C(n, 2)`. A flood of duplicate cables -> large `R` and `S`. All handled by the recurrence and 64-bit counters.

**Complexity.** `O((n + m) * alpha(n))` time, `O(n)` space.

**Code.**

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

    long long redundant = 0;        // cables whose endpoints were already connected
    long long samePairs = 0;        // running number of unordered same-component pairs
    long long prefixRedundantSum = 0; // sum over all queries' answers (running total of redundant so far)

    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        int ru = find(u), rv = find(v);
        if (ru == rv) {
            // both endpoints already connected: this cable is redundant, adds 0 new pairs
            redundant++;
        } else {
            // merging two distinct components of sizes sz[ru], sz[rv]
            long long s1 = sz[ru], s2 = sz[rv];
            samePairs += s1 * s2;   // exactly s1*s2 new unordered cross pairs become same-component
            // union by size
            if (sz[ru] < sz[rv]) swap(ru, rv);
            par[rv] = ru;
            sz[ru] += sz[rv];
        }
        prefixRedundantSum += redundant; // after processing cable e, accumulate current redundant count
    }

    cout << redundant << " " << samePairs << " " << prefixRedundantSum << "\n";
    return 0;
}
```

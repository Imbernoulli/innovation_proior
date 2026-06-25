**Problem.** Split all `n` adventurers (`n <= 16`) into unlabeled teams — a set partition — where every
team has size in `[L, R]` and contains no feuding pair. Count valid partitions modulo `998244353`.
`n = 0` has one (empty) partition; an infeasible size window gives `0`.

**Key idea — anchored subset DP.** Let `f[mask]` be the number of legal *unordered* partitions of the
set `mask`. Precompute `valid[S]` = "`S` is a legal team" (`L <= popcount(S) <= R` and no internal
feud). For a nonempty `mask`, let `low = mask & (-mask)` be its lowest set bit. In any partition, the
element `low` lies in exactly one team `S`, and `S ⊆ mask` contains `low`. Summing over exactly those
teams gives each partition once:

  `f[mask] = sum over legal teams S with low ∈ S ⊆ mask of f[mask \ S]`,  `f[0] = 1`.

Enumerate those `S` by peeling the anchor: `rest = mask ^ low`, run `sub` over all submasks of `rest`,
and form `S = sub | low`. The answer is `f[full] mod p`.

**Why anchoring (the de-dup) is mandatory.** The "obvious" recurrence
`g[mask] = sum over every legal S ⊆ mask of g[mask \ S]` counts **ordered** lists of teams, so a
partition into `k` blocks is counted `k!` times. On `n = 3` with no feuds and `L = 1, R = 3` it returns
`13 = 1!·1 + 2!·3 + 3!·1` (ordered partitions) instead of `B(3) = 5`. Restricting the first team to
contain `low` makes each partition's decomposition unique — no `k!`, no modular inverse, no
post-correction.

**Pitfalls.**
1. *Double-count.* Summing over all submasks (not just anchor-containing ones) over-counts by the block
   orderings. Force `low ∈ S`. This is the single load-bearing line.
2. *Enumerating the right submasks.* To get exactly the subsets of `mask` containing `low`, iterate
   `sub` over submasks of `rest = mask ^ low` and set `S = sub | low`. Iterating submasks of `mask`
   directly and filtering `if (S & low)` also works but is easy to get backwards; peeling the anchor is
   cleaner and never misses the all-anchor team `S = {low}` (the `sub = 0` case).
3. *Off-by-one in the size filter.* The window is inclusive: reject only `popcount < L` or
   `popcount > R`.
4. *Modulus.* Reduce after each addition; the final `% MOD` is a safety net. Counts stay in `[0, p)`,
   so `long long` never overflows.
5. *`n = 0`.* `full = 0`, the loop body never runs, and `f[0] = 1` is the answer — do not special-case
   it to `0`.

**Edge cases.** `n = 0` -> `1`. Single adventurer with `L = 2` -> `0` (no legal team). `L = R = 2` on
odd `n` -> `0` (no perfect pairing). Every pair feuds, `L = 1` -> `1` (all singletons); `L = 2` -> `0`.
No feuds, `L = 1, R = n` -> Bell number `B(n) mod p` (e.g. `n = 16` -> `497698617`).

**Complexity.** `O(2^n · n)` legality precompute plus the classic `O(3^n)` sum-over-submasks DP. At
`n = 16` that is ~`4.3·10^7` operations, ~20 ms and a few MB — well inside 1 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;

int main() {
    int n, L, R, m;
    if (!(cin >> n >> L >> R >> m)) return 0;

    // feud[i] = bitmask of adventurers that i refuses to share a team with.
    vector<int> feud(n, 0);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;             // 0-indexed
        feud[u] |= (1 << v);
        feud[v] |= (1 << u);
    }

    int full = (1 << n) - 1;

    // valid[S] = 1 iff subset S is a legal team:
    //   size in [L,R] and contains no feuding pair.
    vector<char> valid(1 << n, 0);
    for (int S = 1; S <= full; S++) {
        int sz = __builtin_popcount(S);
        if (sz < L || sz > R) { valid[S] = 0; continue; }
        bool ok = true;
        int t = S;
        while (t) {
            int i = __builtin_ctz(t);
            t &= t - 1;
            if (feud[i] & S) { ok = false; break; }   // i feuds with someone also in S
        }
        valid[S] = ok ? 1 : 0;
    }

    // f[mask] = number of partitions of the set "mask" into legal teams (unlabeled).
    vector<long long> f(1 << n, 0);
    f[0] = 1;
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);          // lowest set bit, the anchor element
        long long acc = 0;
        // Enumerate submasks S of mask that CONTAIN the anchor bit "low".
        // This forces each partition to be counted exactly once: the team
        // holding the lowest-indexed element of "mask" is chosen first.
        int rest = mask ^ low;             // bits we may freely add to the anchor team
        for (int sub = rest; ; sub = (sub - 1) & rest) {
            int S = sub | low;             // team containing the anchor
            if (valid[S]) {
                acc += f[mask ^ S];
                if (acc >= MOD) acc -= MOD;
            }
            if (sub == 0) break;
        }
        f[mask] = acc;
    }

    cout << f[full] % MOD << "\n";
    return 0;
}
```

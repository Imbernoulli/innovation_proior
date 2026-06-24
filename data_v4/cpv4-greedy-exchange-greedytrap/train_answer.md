**Problem.** A postage meter stocks `n` distinct stamp denominations `d[0..n-1]` (cents), each in unlimited supply. A parcel needs *exactly* `A` cents. Pick a multiset of stamps (repetition allowed) summing to exactly `A` using as **few stamps** as possible; print that minimum, or `-1` if no multiset of the denominations sums to exactly `A`. Read `n A` then the `n` denominations from stdin; print one integer.

**Why the obvious greedy is wrong.** "Repeatedly take the largest denomination `<= remaining`" fails because hitting an exact target with the fewest pieces is a *global* optimization and greedy commits to the biggest piece with no lookahead. On `d = [1, 3, 4]`, `A = 6`, greedy takes `4 + 1 + 1 = 6`, i.e. `3` stamps, but `3 + 3 = 6` is `2` stamps. (Second witness: `d = [1, 5, 8]`, `A = 10`: greedy `8 + 1 + 1 = 3` vs `5 + 5 = 2`.) Grabbing the large stamp commits to a remainder the denomination set fills poorly. Greedy is optimal only for special "canonical" systems, and the problem promises no canonicity, so it is discarded.

**Key idea — unbounded shortest-combination DP.** Let `dp[v]` = the minimum number of stamps summing to exactly `v`. Any optimal multiset for `v` has a last stamp `d[i] <= v`; removing it leaves an optimal multiset for `v - d[i]`. Hence

- `dp[v] = 1 + min over i with d[i] <= v of dp[v - d[i]]`,
- base case `dp[0] = 0` (empty multiset; this also makes `A = 0` return `0`),
- `v` is unreachable (`dp[v] = INF`) if no legal, reachable predecessor exists.

Fill `v` from `1` to `A`. Answer: `dp[A]`, or `-1` if `dp[A] == INF`.

**Correctness.** Optimal-substructure: stripping one stamp off an optimal solution for `v` yields a feasible solution for `v - d[i]` that must itself be optimal (else swap in a better one and improve `v`, contradiction). Filling amounts in increasing order guarantees every predecessor `v - d[i] < v` is already final when `v` is computed. Unreachable amounts keep the `INF` sentinel and never masquerade as reachable because the relaxation is guarded on `dp[v - d[i]] != INF`.

**Pitfalls.**
1. *Greedy trap.* Largest-first overshoots the stamp count on non-canonical sets (`[1,3,4]@6`, `[1,5,8]@10`). Use the DP, not greedy.
2. *Unguarded relaxation.* Relax only when `d[i] <= v` AND `dp[v - d[i]] != INF`. Dropping the first guard reads `dp[negative]` — out-of-bounds undefined behavior (`[4]@2` -> `dp[-3]`). Dropping the second computes `INF + 1`, fabricating reachability for impossible amounts and risking sentinel overflow.
3. *`A = 0` vs `-1`.* The empty multiset makes `A = 0` answer `0`, never `-1`; the `dp[0] = 0` base case plus the empty loop handle it.
4. *Off-by-one allocation.* Size the table `A + 1` so `dp[A]` is in range; `vector<long long> dp(A)` would read out of bounds at `dp[A]`.

**Edge cases.** `A = 0` -> `0`; parity-unreachable target (all even denominations, odd `A`) -> `-1`; a single denomination not dividing into `A` -> `-1`; a denomination larger than `A` is ignored by the `d[i] <= v` guard; counts never exceed `A <= 10^5`, so no overflow.

**Complexity.** `O(n * A)` time (`<= 10^7` relaxations at the max scale, ~11 ms), `O(A)` memory (~0.8 MB).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long A;
    if (!(cin >> n >> A)) return 0;          // empty input -> nothing to do
    vector<long long> d(n);
    for (auto &x : d) cin >> x;

    // Unbounded "fewest stamps to make exactly A" by DP over amounts 0..A.
    // dp[v] = minimum number of stamps summing to exactly v, or INF if impossible.
    const long long INF = (long long)4e18;
    vector<long long> dp(A + 1, INF);
    dp[0] = 0;
    for (long long v = 1; v <= A; v++) {
        for (int i = 0; i < n; i++) {
            if (d[i] <= v && dp[v - d[i]] != INF) {
                long long cand = dp[v - d[i]] + 1;
                if (cand < dp[v]) dp[v] = cand;
            }
        }
    }

    if (dp[A] == INF) cout << -1 << "\n";
    else cout << dp[A] << "\n";
    return 0;
}
```

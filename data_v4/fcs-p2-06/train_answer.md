**Problem.** A chain of `n` matrices `A[1..n]` is to be multiplied left-to-right, with `A[i]` of
dimension `p[i-1] x p[i]`, so the chain is given by the `n+1` dimensions `p[0..n]`. Multiplying an
`a x b` by a `b x c` matrix costs `a*b*c` scalar multiplications. Associativity fixes the result but
not the cost; choose the parenthesization minimizing the total scalar multiplications and print it.
With `0` or `1` matrices the answer is `0`. Read `n` and the `n+1` dimensions from stdin.

**Why the obvious greedy is wrong.** "At each step multiply the adjacent pair that is cheapest right
now, then collapse it" fails because each multiplication eliminates an interior dimension, and the
locally cheapest merge can expose a huge interior dimension as a boundary that every later step must
pay for. On `p = [10, 1, 100, 10]` greedy multiplies `A[1]*A[2]` (cost `10*1*100 = 1000`) into a
`10x100` matrix, then pays `10*100*10 = 10000` against `A[3]`, totalling `11000`. The optimal order
`A[1]*(A[2]*A[3])` multiplies `A[2]*A[3]` first (`1*100*10 = 1000`, a `1x10` matrix) and then
`10*1*10 = 100`, totalling `1100` — ten times cheaper. Greedy is discarded.

**Key idea — interval DP over sub-chains.** Let `dp[i][j]` be the minimum cost to reduce `A[i..j]` to
one matrix. Every parenthesization performs some multiplication last, splitting `A[i..j]` into
`A[i..k]` (a `p[i-1] x p[k]` matrix) and `A[k+1..j]` (a `p[k] x p[j]` matrix) for one `k` with
`i <= k < j`. So:

- `dp[i][i] = 0`,
- `dp[i][j] = min over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j] )`.

This is exhaustive over all parenthesizations (every full binary tree has a unique root split), and
filling the table in increasing chain length `len = j - i + 1` from `2` to `n` guarantees every
needed sub-chain is already computed. The answer is `dp[1][n]`. Complexity: `O(n^2)` intervals times
`O(n)` splits = `O(n^3)`; at `n = 300` that is `~2.7*10^7` operations, well under the time limit.

**Two pitfalls to get right.**
1. *Seeding the minimum.* Initialize the per-interval running minimum `best` to a large sentinel
   (`INF = LLONG_MAX/4`), not `0`. Every candidate cost is positive, so seeding with `0` makes the
   comparison `cost < best` never fire and reports every multi-matrix chain as free. (A trace of
   `[2, 3, 4]` returning `0` instead of `24` exposes exactly this.) The sentinel is only ever read
   inside a comparison, never has a cost added to it, so it cannot overflow.
2. *Overflow.* With `n` up to `300` and dimensions up to `1000`, the total can reach `~3*10^11`
   (e.g. `[1000]*301 -> 299000000000`); use `long long`. An `int` is a silent wrong-answer on large
   tests.

**Edge cases.** `n = 0` and `n = 1` are peeled off before the table fill and return `0` (nothing to
multiply); `n = 2` reduces to the single product `p[0]*p[1]*p[2]`.

**Verification.** Differential-tested against an independent memoized-recursion oracle and a pure
no-memoization parenthesization-enumeration oracle over 1100+ random, edge, and greedy-trap cases
with zero mismatches; the `n = 300` worst case runs in milliseconds.

**Complexity.** `O(n^3)` time, `O(n^2)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (no matrices) -> answer 0
    vector<long long> p(n + 1);
    for (auto &x : p) cin >> x;            // p[0..n]: matrix i is p[i-1] x p[i]

    if (n <= 1) { cout << 0 << "\n"; return 0; } // 0 or 1 matrix: nothing to multiply

    // dp[i][j] = min scalar multiplications to multiply matrices i..j (1-indexed, inclusive).
    // dp[i][i] = 0; for i < j, split at k in [i, j-1]:
    //   dp[i][j] = min over k of dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j].
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, 0));

    for (int len = 2; len <= n; len++) {        // chain length
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            long long best = INF;
            for (int k = i; k < j; k++) {
                long long cost = dp[i][k] + dp[k + 1][j]
                                 + p[i - 1] * p[k] * p[j];
                if (cost < best) best = cost;
            }
            dp[i][j] = best;
        }
    }

    cout << dp[1][n] << "\n";
    return 0;
}
```

**Problem.** A hallway is a row of `n` panels with roughness `a[1..n]` (`a[i] >= 1`). Cover it
completely with runner rugs; each rug is an **inclusive** interval `[l, r]` of panels with length
`r - l + 1 <= L`, costing `K + max(a[l..r])`. The rugs partition `1..n` into consecutive blocks.
Minimize the total cost. Read `n, K, L` and `a[1..n]` from stdin; print the minimum.

**Why the obvious greedy is wrong.** Because a rug is charged only for its single roughest panel,
"extend the current rug as far as legal" is globally wrong. On `a = [1, 5, 5, 1, 5]`, `K = 2`,
`L = 2`, greedy-longest lays `[1,2],[3,4],[5,5]` for `7 + 7 + 7 = 21`, but isolating the cheap first
panel — `[1,1],[2,3],[4,5]` — costs `3 + 7 + 7 = 17`. Extending the first rug forced a smooth panel
to share a rug with a rough one and pay `5` instead of `1`. Greedy is discarded.

**Key idea — partition DP over a prefix.** Let `dp[i]` = minimum cost to cover the first `i` panels
(panels `1..i`), with `dp[0] = 0`. The last rug ends at panel `i` and starts at panel `j+1`, so it
covers the inclusive interval `[j+1, i]` of length `i - j`. The legality `1 <= i - j <= L` means `j`
ranges over `[max(0, i - L), i - 1]`. Sweeping `j` downward from `i-1`, maintain a running
`curMax = max(curMax, a[j+1])` (the rug's leftmost panel as it grows) and take

  `dp[i] = min over j of  dp[j] + K + curMax`.

Answer: `dp[n]`. Complexity `O(n * L)`.

**Pitfalls — all on the inclusive boundary.**
1. *Length cap off-by-one.* The rug `[j+1, i]` has length `i - j`, so the smallest legal `j` is
   `i - L`, i.e. loop `j >= max(0, i - L)`. Looping to `lo - 1 = i - L - 1` admits a rug of length
   `L + 1`. Trace `a = [1,2,2]`, `K = 1`, `L = 2`: the bad bound lets a single rug cover `[1,3]`
   (length 3) for cost `3`, beating the legal answer `5`.
2. *Wrong max endpoint.* The running max must fold in `a[j+1]` (the rug's inclusive left panel), not
   `a[j]`. With `a[j]` and `j = 0`, a length-1 rug `[1,1]` reads the dummy slot `a[0]` and is priced
   at `K + 0` instead of `K + a[1]`. On the same `[1,2,2]` that undercounts to `4 < 5`.
3. *Overflow.* Up to `n` rugs with `K, a[i] <= 10^9` gives totals near `10^13`; `dp` must be
   `long long`. With large `K` the answer already exceeds 32-bit `int`, so this is a silent
   wrong-answer otherwise.

**Edge cases.** `n = 1` -> the single rug `[1,1]`, cost `K + a[1]`. `L = 1` -> forced singletons,
total `n*K + sum a[i]`. `L = n` -> one rug `[1,n]` is allowed and considered. `L = n - 1` -> the
full-hallway rug is forbidden by exactly the length boundary, so the DP uses at least two rugs (this
is what the off-by-one fix guarantees). `K = 0` -> recurrence unchanged. The `INF` guard
(`dp[j] != INF`) keeps unreachable states from polluting the minimum.

**Complexity.** `O(n * L)` time (at most `2.5*10^7` for `n = L = 5000`), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L;
    if (!(cin >> n >> K >> L)) return 0;      // empty input
    vector<long long> a(n + 1);               // 1-indexed heights a[1..n]
    for (int i = 1; i <= n; i++) cin >> a[i];

    // dp[i] = minimum total cost to tile the first i panels (panels 1..i).
    // dp[0] = 0 (no panels, no rugs). The last rug covers an INCLUSIVE
    // interval [j+1, i] of length (i - (j+1) + 1) = i - j panels, which must
    // be between 1 and L. Its cost is K + max(a[j+1..i]).
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;

    for (int i = 1; i <= n; i++) {
        long long curMax = 0;                 // max over the growing suffix
        // j is the index BEFORE the last rug; last rug = [j+1, i].
        // length = i - j must satisfy 1 <= i - j <= L, so j ranges in
        // [max(0, i - L), i - 1]. We extend the rug leftward from j = i-1.
        int lo = max(0, i - L);
        for (int j = i - 1; j >= lo; j--) {
            curMax = max(curMax, a[j + 1]);   // include panel (j+1) inclusively
            if (dp[j] != INF) {
                long long cand = dp[j] + K + curMax;
                if (cand < dp[i]) dp[i] = cand;
            }
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
```

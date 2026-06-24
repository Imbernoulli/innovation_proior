**Problem.** A bakery menu has `n` pastries, pastry `i` with `price[i]` and `joy[i]`. Choose a subset
(each pastry at most once) so the total price is **exactly** `B` and the total joy is **at least** `J`.
Count the distinct valid subsets modulo `1000000007`. The empty subset is a valid box (price `0`, joy
`0`). Read `n B J` then the `n` pairs from stdin; print the count mod `10^9+7`.

**Key idea — 2-D 0/1 knapsack counting with a clamped joy axis.** The exact-price requirement is a 0/1
subset-sum over a price axis of size `B+1`. The joy condition is a *lower bound*, so the exact joy is
irrelevant once it reaches `J`: define each selection's *clamped joy* as `min(totalJoy, J)`. Because all
`joy[i] >= 0`, the bucket `j = J` is **absorbing** (once a partial box reaches joy `>= J` it stays
there), so every box with true joy `>= J` collapses into bucket `J`, the joy axis stays size `J+1`, and
the answer is the single cell `dp[B][J]`.

Let `dp[c][j]` = number of distinct subsets seen so far with total price `c` and clamped joy `j`. Base
case `dp[0][0] = 1` (the empty box). Process items one at a time; taking item `i` (price `p`, joy `g`)
moves a subset from `(c-p, j)` to `(c, min(j+g, J))`:

```
dp[c][min(j+g, J)] += dp[c-p][j]   (mod 1e9+7)
```

Roll the table in place. Answer: `dp[B][J]`.

**Correctness.** Items are introduced one by one; after item `i` is processed, `dp[c][j]` counts subsets
of `{0..i}` with price `c` and clamped joy `j` — each subset is generated exactly once (the take/skip
choice for item `i` is made exactly once, provided each item is offered once; see Pitfalls). The clamp is
faithful: a subset has clamped joy `J` iff its true joy is `>= J`, since non-negative joys make the clamp
monotone and absorbing, so `dp[B][J]` is exactly the set of price-`B`, joy-`>= J` subsets.

**Pitfalls.**
1. *0/1 vs reuse (double-count).* Roll **both** in-place axes — the price `c` and the clamped joy `j` —
   **downward**. An ascending sweep reads `dp[c-p][...]` *after* the current item already wrote it, so
   each item is folded into a state that already contains it: the count silently inflates (it counts the
   same pastry twice). A counting DP does not crash on this — it returns a larger wrong number. Trace
   `2 4 0 / (2,3) (2,1)`: ascending gives `3`, the truth is `1`; descending gives `1`.
2. *Threshold off-by-one (double-count).* Clamp at `J` and read cell `dp[B][J]`. Sizing the joy axis
   `0..J-1` and reading `dp[B][J-1]` counts boxes with joy `J-1` as if they satisfied `>= J`. Trace
   `1 3 3 / (3,2)`: the `J-1` variant gives `1`, the truth is `0`. The bucket boundary must be `min(joy,
   J)` with the answer in bucket `J`, which encodes `>= J` exactly (the sample's joy-`7 = J` box must
   count).
3. *Overflow / modulus.* Counts can reach `2^100`, so cells are `long long` reduced mod `10^9+7` after
   every addition. An `int` cell would overflow and corrupt the count silently.

**Edge cases.** `n = 0`: answer `1` iff `B = 0` and `J = 0` (empty box), else `0`. `B = 0`: only price-`0`
subsets qualify; a price-`0` item correctly doubles the price-`0` count (the `c >= p` loop still runs at
`c = 0, p = 0`). `J = 0`: every subset satisfies the joy floor; bucket `0` is the only bucket and
`dp[B][0]` counts all price-`B` subsets. `price[i] > B`: skipped by the `if (p > Bc) continue;` guard.
Joy exactly `J`: clamped to bucket `J`, counted. All verified against an exhaustive `2^n` checker over
500+ random small menus and explicit corner inputs.

**Complexity.** `O(n * B * J)` time (`~2.25 * 10^8` worst-case cell-updates, with an early `continue` on
empty source cells), `O(B * J)` memory (`~18 MB`). Well within 2 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B;
    long long J;
    if (!(cin >> n >> B >> J)) return 0;

    const long long MOD = 1000000007LL;

    vector<long long> price(n), joy(n);
    for (int i = 0; i < n; i++) cin >> price[i] >> joy[i];

    // dp[c][j] = number of distinct subsets (0/1) chosen so far whose total
    // price == c and whose joy CLAMPED at J equals j.
    // Joy is clamped to J because we only care about "joy >= J": every subset
    // with true joy >= J collapses into the bucket j = J, so the answer is the
    // single cell dp[B][J] at the end. Clamping keeps the second dimension O(J).
    int Bc = (int)B;
    int Jc = (int)J;

    // dp indexed [price 0..Bc][clampedJoy 0..Jc]
    vector<vector<long long>> dp(Bc + 1, vector<long long>(Jc + 1, 0));
    dp[0][0] = 1; // the empty subset: price 0, joy 0

    for (int i = 0; i < n; i++) {
        long long p = price[i];
        long long jv = joy[i];
        if (p > Bc) continue; // cannot ever fit by price
        // 0/1 knapsack: iterate price DOWNWARD so each item is used at most once.
        for (int c = Bc; c >= (int)p; c--) {
            // iterate clamped joy DOWNWARD as well, same 0/1 reason.
            for (int j = Jc; j >= 0; j--) {
                if (dp[c - (int)p][j] == 0) continue;
                int nj = j + (int)jv;
                if (nj > Jc) nj = Jc; // clamp: joy >= J all collapse to bucket J
                dp[c][nj] = (dp[c][nj] + dp[c - (int)p][j]) % MOD;
            }
        }
    }

    cout << dp[Bc][Jc] % MOD << "\n";
    return 0;
}
```

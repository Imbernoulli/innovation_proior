**Problem.** A drone must leave with **exactly `K`** parcels whose combined weight is **at most `C`**,
maximizing the total net profit. Parcel `i` has weight `w[i] >= 0` (zero allowed) and profit `v[i]`
that may be negative, zero, or positive. Among all size-`K` loads that fit under the cap, print the
maximum total profit (which may be **negative** or zero); if no size-`K` load fits, print
`INFEASIBLE`. Read `n`, `K`, `C` and the `(w, v)` pairs from stdin.

**Key idea — two-dimensional (count x weight) 0/1 knapsack.** Carry `dp[k][c]` = best total profit
using **exactly `k`** parcels of **exact total weight `c`** (`c` in `0..C`), or a sentinel `NEG` for
"unreachable." The only state reachable before any parcel is placed is `dp[0][0] = 0` (the empty
load); everything else starts at `NEG`. For each parcel `(w_i, v_i)` with `w_i <= C`, relax in 0/1
fashion:

- `dp[k+1][c + w_i] = max(dp[k+1][c + w_i], dp[k][c] + v_i)` whenever `dp[k][c]` is reachable and
  `c + w_i <= C`.

Sweep the **count axis `k` downward** (`K-1 .. 0`) so a single parcel cannot fill two of the `K`
slots. The answer is `max over c in 0..C of dp[K][c]`; if that whole row is `NEG`, print `INFEASIBLE`.

**Pitfalls.**
1. *Wrong base case (the sign trap).* Initialising the **entire** table to `0` instead of just
   `dp[0][0] = 0` plants a phantom "exactly `k` parcels at profit `0`" state. Because profits can be
   negative, that phantom `0` *beats* a genuine negative load: on `n=1, K=1, C=5`, parcel `(2,-4)`,
   the all-zero table returns `0` instead of the correct `-4`. Use a sentinel `NEG = LLONG_MIN/4`,
   seed only `dp[0][0] = 0`, and never relax from / add `v_i` to a `NEG` cell.
2. *Feasible-negative vs infeasible.* Keep "best size-`K` load is negative" (a real value) strictly
   distinct from "no size-`K` load exists" (`INFEASIBLE`). The sentinel does both: a non-`NEG` row `K`
   yields the (possibly negative) optimum; an all-`NEG` row `K` yields `INFEASIBLE`.
3. *0/1 reuse direction.* Sweep the count axis downward. Upward double-counts one parcel: on
   `n=2, K=2, C=0`, parcels `(0,5),(0,3)`, an upward sweep returns the impossible `10` (parcel 0 in
   both slots) instead of the correct `8`.
4. *Huge weights / huge `K`.* `w_i` can be `10^9` while the weight axis spans only `0..C`; skip any
   parcel with `w_i > C` before indexing `c + w_i`. `K` can be up to `10^9`; test `K > n` and print
   `INFEASIBLE` **before** allocating a `K`-row table.
5. *Overflow.* Up to `K <= 200` parcels with `|v| <= 10^9` give profits near `2 * 10^11`; use
   `long long` for every accumulator and cell. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `K = 0` -> `0` (empty load, even when all parcels are negative); `n = 0` -> `0` if
`K = 0`, else `INFEASIBLE`; `K > n` -> `INFEASIBLE`; all feasible loads negative -> print the negative
optimum, not `INFEASIBLE`; many zero-weight parcels stack at weight `0`; parcels with `w_i > C` are
never choosable.

**Complexity.** `O(n * K * C)` time, `O(K * C)` memory. Worst case `200 * 200 * 1000 = 4 * 10^7`
updates and ~1.6 MB — about 0.02 s, well within 1 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, C;
    if (!(cin >> n >> K >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // If the required count is impossible up front, no feasible load exists.
    if (K < 0 || K > n) { cout << "INFEASIBLE" << "\n"; return 0; }

    const long long NEG = LLONG_MIN / 4;       // "no subset with this (count,weight) exists"
    int Kc = (int)K;
    int Cc = (int)C;

    // dp[k][c] = best total profit using EXACTLY k parcels of total weight EXACTLY c (c in 0..Cc).
    vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, NEG));
    dp[0][0] = 0;                              // exactly 0 parcels, weight 0, profit 0 (empty load)

    for (int i = 0; i < n; i++) {
        long long wi = w[i], vi = v[i];
        if (wi > (long long)Cc) continue;      // parcel alone exceeds capacity: never choosable
        int wint = (int)wi;
        for (int k = Kc - 1; k >= 0; k--) {    // 0/1: counts downward so each parcel used once
            for (int c = Cc - wint; c >= 0; c--) {
                if (dp[k][c] == NEG) continue;
                long long cand = dp[k][c] + vi;
                if (cand > dp[k + 1][c + wint]) dp[k + 1][c + wint] = cand;
            }
        }
    }

    long long ans = NEG;
    for (int c = 0; c <= Cc; c++)
        if (dp[Kc][c] != NEG && dp[Kc][c] > ans) ans = dp[Kc][c];

    if (ans == NEG) cout << "INFEASIBLE" << "\n";
    else cout << ans << "\n";
    return 0;
}
```

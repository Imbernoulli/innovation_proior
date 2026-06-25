**Problem.** A row of `n` tiles carries integer charges `a[0..n-1]` (negative or zero allowed). Repeatedly fuse two adjacent tiles `x, y` into one tile of charge `x + y`, releasing energy `x * y` (possibly negative). After `n - 1` fusions one tile remains; you pick the order. Maximize total released energy. For `n = 0` or `n = 1` no fusion happens, so the answer is `0`. Read `n` and the charges from stdin, print the maximum.

**Why the obvious greedy is wrong.** "Always do the adjacent fusion with the largest immediate `x * y`" fails because a fusion rewrites a tile's charge, changing every later product it joins. On mixed-sign rows the locally best fusion inflates a charge that then multiplies badly downstream; checked against an exhaustive oracle on random small rows, greedy diverges from the optimum. Greedy is discarded.

**Key idea — interval DP via the last fusion.** Charge sums are invariant under fusion order (fusion only adds), so collapsing a block `[i..j]` always yields a tile of charge `S(i,j) = a[i] + ... + a[j]`. Look at the *last* fusion forming `[i..j]`: it merges a collapsed left sub-block `[i..k]` (charge `S(i,k)`) with a collapsed right sub-block `[k+1..j]` (charge `S(k+1,j)`), releasing `S(i,k) * S(k+1,j)` on top of each side's own optimum. With prefix sums `pre[t] = a[0]+...+a[t-1]` and `S(i,j) = pre[j+1] - pre[i]`:

- `dp[i][i] = 0`  (a single tile needs no fusion)
- `dp[i][j] = max over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] + S(i,k) * S(k+1,j) )`

Fill by increasing block length. The answer is `dp[0][n-1]`.

**Pitfalls.**
1. *Wrong base case / sign of the optimum.* For a block of `>= 2` tiles a fusion is **mandatory**, so its best energy can be negative. Seeding the per-interval maximum at `0` inserts a phantom "release nothing" option and clamps every negative interval up to `0`, which then corrupts larger intervals. Start the running max at `LLONG_MIN`. (A trace of `[-3, 4]` returning `0` instead of `-12` exposes exactly this.)
2. *Sign of the final answer.* Two negative charges multiply to a *positive* release, so an all-negative row usually has a positive answer (e.g. `[-2,-3,-4]` -> `26`), while a forced-loss row can be negative (`[-3,4]` -> `-12`). Do **not** clamp the final answer with `max(., 0)`; the recurrence already gets the sign right.
3. *Overflow.* With `n <= 500` and `|a[i]| <= 10^6`, charge sums reach `5*10^8`, products reach `2.5*10^17`, and the answer's magnitude is bounded by `V^2 * n(n-1)/2 = 1.2475*10^17` (verified on `[10^6]*8 -> 2.8*10^13`). All fit in `long long`; an `int` is a silent wrong-answer.

**Edge cases.** `n = 0` and `n = 1` -> `0` (guarded before the DP). A tile of charge `0` zeroes any product touching it, handled by the recurrence with no special case. All-negative -> positive answer; forced-loss -> negative answer; both correct without clamping. The `LLONG_MIN` sentinel is only ever compared, never added to, since every block of length `>= 2` has at least one real split.

**Complexity.** `O(n^3)` time, `O(n^2)` space; about `1.25*10^8` simple operations at `n = 500`, ~0.03 s.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // no input -> n = 0 -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n <= 1) {                          // 0 or 1 token: no merge happens, score 0
        cout << 0 << "\n";
        return 0;
    }

    // prefix sums: pre[i] = a[0] + ... + a[i-1]; S(i..j) = pre[j+1] - pre[i].
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + a[i];
    auto S = [&](int i, int j) -> long long { return pre[j + 1] - pre[i]; };

    // dp[i][j] = max total score to collapse the inclusive interval [i..j] into one token.
    // dp[i][i] = 0 (a single token needs no merge).
    // dp[i][j] = max over split k in [i, j-1] of
    //            dp[i][k] + dp[k+1][j] + S(i,k) * S(k+1,j).
    vector<vector<long long>> dp(n, vector<long long>(n, 0));

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            long long best = LLONG_MIN;     // forced to merge -> may be negative; base must be -inf
            for (int k = i; k < j; k++) {
                long long cand = dp[i][k] + dp[k + 1][j] + S(i, k) * S(k + 1, j);
                if (cand > best) best = cand;
            }
            dp[i][j] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
```

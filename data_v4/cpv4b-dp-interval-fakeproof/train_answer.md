**Problem.** `n` stones in a row carry values `a[0..n-1]` with `0 <= a[i] < 2^20`. A merge takes two
*currently adjacent* stones `x` (left), `y` (right), replaces them by one stone valued `x XOR y`, and
costs `x OR y` on the two values at the moment they merge. After `n - 1` merges one stone remains;
output the minimum possible total cost. Read `n` then the `n` values from stdin; print the minimum.
For `n <= 1` the cost is `0`.

**The trap, and why the answer is not a closed form.** XOR is associative and commutative, so the
final stone always equals `a[0] ^ ... ^ a[n-1]` regardless of merge order. It is tempting to conclude
the *cost* is likewise order-independent and write a one-line formula. Do not assert one — derive a
candidate and check it. Two natural guesses on the sample `[6, 5, 3]` (true answer `10`): the global
OR is `6|5|3 = 7`, so `(n-1)*OR = 2*7 = 14`; and the sum of adjacent-pair ORs is `(6|5)+(5|3) = 14`.
Both give `14 != 10`, so both are false. The reason: a merge's cost is the OR of two *partial* XORs,
and partial XORs cancel bits the global OR keeps (merging `6` and `5` into `6^5 = 3` kills the high
bit, making the next merge `3|3 = 3` cheap). No order-independent quantity sees that cancellation, so
order matters and an interval DP is required.

**Key idea — interval DP over contiguous blocks.** At any moment the surviving stones partition
`[0, n-1]` into contiguous segments, and a stone over `[l..r]` has value `seg(l,r) = XOR of a[l..r]`.
Get that in `O(1)` from a prefix-XOR array: `px[0]=0`, `px[i+1]=px[i]^a[i]`, `seg(l,r)=px[r+1]^px[l]`.
Let `dp[l][r]` be the minimum cost to merge `[l..r]` to one stone. The last merge joins a non-empty
left block `[l..k]` and non-empty right block `[k+1..r]`:

- `dp[l][l] = 0` (one stone, no merges);
- `dp[l][r] = min over k in [l, r-1] of ( dp[l][k] + dp[k+1][r] + ( seg(l,k) | seg(k+1,r) ) )`.

Answer: `dp[0][n-1]`. Hand-check on `[6,5,3]`: `dp[0][1]=6|5=7`, `dp[1][2]=5|3=7`; then split `k=0`
gives `7+(6|6)=13`, split `k=1` gives `7+(3|3)=10`, so `dp[0][2]=10` — matching the two real merge
orders.

**Pitfalls.**
1. *False bitwise identity.* The order-independence of the final value does **not** make the cost a
   closed form. Always refute the guessed formula on a concrete small case before trusting it; here
   both natural guesses give `14` against the true `10`.
2. *Split-loop off-by-one.* `k` must run `l .. r-1`, never `k = r`. With `k = r` the right block is
   empty, `dp[r+1][r]` is meaningless, and you read the cell `dp[l][r]` you are still filling.
3. *Base case and sentinel.* A zero-initialized `dp` makes the diagonal `dp[l][l] = 0` correct for
   free; the `INF = LLONG_MAX/4` seed of `best` is always overwritten (a `len>=2` segment has at
   least split `k=l`) and never enters an addition, so it cannot overflow.
4. *Overflow.* Total cost `< 1500 * (2^20-1) < 1.6*10^9`; use `long long`.

**Edge cases.** `n = 0` -> `0` (explicit guard); `n = 1` -> `0` (length loop never runs);
`n = 2` -> the single forced `a[0] | a[1]`; all-zero row -> `0`; rows built to punish a greedy
"merge the cheapest adjacent pair now" order are handled because every split is considered.

**Complexity.** `O(n^2)` states times `O(n)` per state = `O(n^3)` time, `O(n^2)` memory. For `n=1500`
that is about `0.8 s` and `~21 MB` — inside a 2 s / 256 MB budget.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // px[i] = XOR of a[0..i-1]; XOR of segment [l..r] = px[r+1] ^ px[l].
    vector<long long> px(n + 1, 0);
    for (int i = 0; i < n; i++) px[i + 1] = px[i] ^ a[i];

    auto segxor = [&](int l, int r) -> long long { return px[r + 1] ^ px[l]; };

    const long long INF = LLONG_MAX / 4;
    // dp[l][r] = minimal total cost to merge segment [l..r] into one stone.
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    for (int len = 2; len <= n; len++) {
        for (int l = 0; l + len - 1 < n; l++) {
            int r = l + len - 1;
            long long best = INF;
            for (int k = l; k < r; k++) {
                // left part [l..k] -> value segxor(l,k); right part [k+1..r] -> value segxor(k+1,r)
                long long cost = dp[l][k] + dp[k + 1][r]
                               + (segxor(l, k) | segxor(k + 1, r));
                best = min(best, cost);
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
```

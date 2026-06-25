**Problem.** Given `n` stamp denominations, denomination `i` having positive value `v[i]` and supply `c[i]`, count the number of **distinct multisets** of stamps — at most `c[i]` of denomination `i` — whose face values sum to **exactly** `S`, reported modulo `MOD`. Two payments are the same iff they use the same count of every denomination; order never matters. Read `n S MOD` and the `n` pairs from stdin, print the count mod `MOD`.

**Why the obvious capacity-outer DP is wrong.** The four-line recurrence `f[0]=1; f[s] = sum over denominations j of f[s - v[j]]` counts *ordered* compositions, not multisets. On `S = 5` with values `{1, 2}` it returns `8` (a Fibonacci count of sequences of 1s and 2s), but only `3` distinct multisets exist: `{1,1,1,1,1}`, `{1,1,1,2}`, `{1,2,2}`. Reaching a sum by "adding 2 then 3" and "adding 3 then 2" is the same multiset but two paths in that DP. Discard it.

**Key idea — denomination-outer DP with a per-residue sliding window.** Process denominations one at a time, keeping `dp[s]` = number of multisets over the denominations seen so far that sum to `s`. Bringing in denomination `i` (value `val`, supply `lim`):

`dp_new[s] = sum_{k=0}^{lim} dp_old[s - k*val]`  (use `k` copies; remainder is a multiset over earlier denominations).

Because each denomination is committed once in a fixed order, every multiset is produced exactly once. The bounded sum runs over the indices `s, s-val, s-2*val, ...` — a window of `lim+1` consecutive terms within the residue class `r = s mod val`. So for each residue `r` sweep `s = r, r+val, ...`, maintain a running `window`, add `dp_old[s]`, and once the window holds more than `lim+1` terms drop the one `lim+1` steps back (`dp_old[s - (lim+1)*val]`). That makes each denomination `O(S)`, total `O(n*S)`.

**Pitfalls to get right.**
1. *Loop order is the dedup.* Denomination outer, capacity inner counts multisets; the reverse counts permutations. This is the whole problem.
2. *Window-width off-by-one.* `k` ranges over `0..lim`, which is `lim+1` values, so the window holds `lim+1` terms — evict only when `count > lim + 1`, and drop index `s - (lim+1)*val`. Using `count > lim` silently drops the zero-copies term, under-counting (a mis-dedup of copies). A trace of `dp=[1,1,1], val=1, lim=1` returning `[1,1,1]` instead of `[1,2,2]` exposes exactly this.
3. *Large `val` and the residue loop.* Bound it by `r < val && r <= S`; otherwise `val` up to `10^9` spins the loop a billion times. Residues `r > S` index nothing.
4. *Overflow and `MOD`.* Keep residues in `long long`, only ever *add* two values `< MOD` (never multiply), reduce immediately. Write `dp[0] = 1 % MOD` so `MOD = 1` yields `0`.

**Edge cases.** `S = 0` -> `1` (empty combination). Unreachable target -> `0`. A denomination with `v[i] > S` is forced to zero copies (handled by `r <= S`). Very large `c[i]` makes the window never evict, i.e. effectively unbounded — correct. `MOD = 1` -> `0`. Non-prime `MOD` is fine because the method never divides.

**Complexity.** `O(n*S)` time, `O(S)` extra space. At `n = 200, S = 2*10^5` this is `~4*10^7` operations, well under a second.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    long long MOD;
    if (!(cin >> n >> S >> MOD)) return 0;

    vector<long long> v(n), c(n);
    for (int i = 0; i < n; i++) cin >> v[i] >> c[i];

    // dp[s] = number of distinct multisets (combinations) of stamps, drawn from
    // the denominations processed so far, whose values sum to exactly s, mod MOD.
    // Denomination in the OUTER loop, capacity in the INNER loop: each multiset
    // is counted exactly once (unordered).
    vector<long long> dp(S + 1, 0);
    dp[0] = 1 % MOD;

    for (int i = 0; i < n; i++) {
        long long val = v[i];
        long long lim = c[i];               // max copies of denomination i

        // ndp[s] = sum_{k=0..lim} dp[s - k*val]  (k copies of this denomination,
        // remainder a combination over the PREVIOUS denominations). We compute it
        // with a sliding window of width (lim+1) along each residue class mod val,
        // so the transition is O(S) rather than O(S * lim).
        vector<long long> ndp(S + 1, 0);
        for (long long r = 0; r < val && r <= S; r++) {
            long long window = 0;           // sum of dp at the last (lim+1) terms
            long long s = r;
            long long count = 0;            // how many terms are currently in window
            for (; s <= S; s += val) {
                window += dp[s];
                if (window >= MOD) window -= MOD;
                count++;
                if (count > lim + 1) {      // window holds more than lim+1 terms: drop oldest
                    long long old = s - (lim + 1) * val;
                    window -= dp[old];
                    if (window < 0) window += MOD;
                    count--;
                }
                ndp[s] = window;
            }
        }
        dp.swap(ndp);
    }

    cout << dp[S] % MOD << "\n";
    return 0;
}
```

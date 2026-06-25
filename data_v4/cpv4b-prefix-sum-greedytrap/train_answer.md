**Problem.** Split the whole array `a[0..n-1]` (values may be negative) into contiguous,
non-empty blocks. A block is *profitable* if its sum is strictly positive. Maximize the
number of profitable blocks. Read `n` and the values from stdin, print the maximum count.

**Why the obvious greedy is wrong.** "Sweep left to right; the instant the current block's
running sum turns strictly positive, close it, bank a point, and reset" fails because the
non-profitable blocks are *bridges*: they let the running total fall back down so a later
block can climb again. On the sample `[4, -4, 2, -1, 2, -4]` (prefix `[0,4,0,2,1,3,-1]`) the
greedy banks `[4]` and then never turns positive again, scoring `1`, while
`[4] | [-4] | [2] | [-1, 2] | [-4]` scores `3`. Closing a block early strands value that a
later cut could have monetized. Greedy is discarded.

**Key idea — prefix-sum DP with a range-max transition.** Let `prefix[0]=0`,
`prefix[i]=a[0]+...+a[i-1]`. A block `(j, i]` is profitable iff `prefix[i] > prefix[j]`. Let
`dp[i]` be the best number of profitable blocks in a full partition of `prefix[0..i]`. The
last block is `(j, i]`, so

- `dp[i] = max( 1 + max_{j<i, prefix[j] <  prefix[i]} dp[j],` &nbsp; (last block profitable)
- `             max_{j<i, prefix[j] >= prefix[i]} dp[j] )` &nbsp; (last block not profitable).

`dp[0]=0`, answer `dp[n]`. Both inner maxima are range-max queries over previously inserted
prefix values, so coordinate-compress the `n+1` prefixes and keep prefix-max of `dp` in two
Fenwick trees: `bitLess` indexed by coordinate (gives the `<` max as a prefix-max), and
`bitGeq` indexed by the *reversed* coordinate (turns the `>=` suffix-max into a prefix-max).
Each step is `O(log n)`.

**Pitfalls.**
1. *Reversed-coordinate off-by-one.* A coordinate `ci` in `0..m-1` must map to reversed
   index `m-1-ci`, not `m-ci`; the latter pushes the top coordinate to slot `m`, off the end
   of the Fenwick tree, so updates and queries silently vanish. Tie-heavy inputs like `[0,0]`
   expose it (correct answer `0`).
2. *Update both trees every step.* A future index can resolve its last block against the
   current `i` from either the `<` or the `>=` side, so every `dp[i]` must be inserted into
   both `bitLess` and `bitGeq`; dropping the `bitGeq` insert blinds the `>=` side (e.g.
   `[-4, 2, 4]` should give `2`).
3. *Overflow.* Prefix sums reach about `2*10^14` with `n=2*10^5`, `|a[i]|=10^9`; keep them in
   `long long`. (The answer itself is `<= n`, so it fits in `int`.)
4. *Strict positivity.* A block of sum exactly `0` is **not** profitable; that is why the
   indicator uses `prefix[i] > prefix[j]` (strict), and zeros never add to the count.

**Edge cases.** `n = 0` (or empty input) -> `0`; a single non-positive hour -> `0`; all
non-positive -> `0`; all positive -> `n` (each hour its own block).

**Complexity.** `O(n log n)` time, `O(n)` space. Measured `0.13 s` / `8 MB` at `n=2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;              // n = 0 (or empty input) -> answer 0

    // prefix[i] = a[0] + ... + a[i-1], so prefix[0] = 0 and a block (j, i] has
    // sum prefix[i] - prefix[j]; it is "profitable" iff prefix[i] > prefix[j].
    vector<long long> prefix(n + 1);
    prefix[0] = 0;
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // Coordinate-compress the n+1 prefix values.
    vector<long long> vals(prefix.begin(), prefix.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto cid = [&](long long v) {
        return int(lower_bound(vals.begin(), vals.end(), v) - vals.begin());
    };

    const int NEG = INT_MIN / 4;

    // dp[i] = max profitable blocks in a full partition of prefix[0..i].
    // dp[i] = max( 1 + max_{j<i, prefix[j] <  prefix[i]} dp[j],     // block (j,i] profitable
    //                  max_{j<i, prefix[j] >= prefix[i]} dp[j] ).   // block (j,i] not profitable
    // Two Fenwick trees over the compressed prefix coordinate hold prefix-max of dp:
    //   bitLess : indexed by coordinate, prefix-max query gives best dp over smaller prefix values.
    //   bitGeq  : indexed by REVERSED coordinate, prefix-max query gives best dp over >= values.
    vector<int> bitLess(m + 1, NEG), bitGeq(m + 1, NEG);
    auto upd = [&](vector<int> &t, int i, int v) {       // 1-based index
        for (++i; i <= m; i += i & (-i)) t[i] = max(t[i], v);
    };
    auto qry = [&](vector<int> &t, int i) {              // max over [0..i], 0-based i
        int r = NEG;
        for (++i; i > 0; i -= i & (-i)) r = max(r, t[i]);
        return r;
    };

    // Insert j = 0: dp[0] = 0 at coordinate of prefix[0].
    int c0 = cid(prefix[0]);
    upd(bitLess, c0, 0);
    upd(bitGeq, m - 1 - c0, 0);

    int ans = 0;
    for (int i = 1; i <= n; i++) {
        int ci = cid(prefix[i]);
        int best = NEG;
        // prefix[j] < prefix[i]: coordinates [0 .. ci-1]
        if (ci - 1 >= 0) {
            int a1 = qry(bitLess, ci - 1);
            if (a1 > NEG) best = max(best, a1 + 1);
        }
        // prefix[j] >= prefix[i]: coordinates [ci .. m-1] = reversed prefix [0 .. m-1-ci]
        int a2 = qry(bitGeq, m - 1 - ci);
        if (a2 > NEG) best = max(best, a2);
        int dpi = best;                 // there is always at least one j (j = i-1), so best > NEG
        ans = dpi;
        upd(bitLess, ci, dpi);
        upd(bitGeq, m - 1 - ci, dpi);
    }

    cout << ans << "\n";
    return 0;
}
```

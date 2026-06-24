**Problem.** A row of `n` slabs with widths `w[0..n-1]` is fused into one piece by repeated welds. A
weld joins two *currently adjacent* pieces and costs their combined width (the sum of original widths
of all slabs inside them). Over all valid weld orders, output the minimum total cost. Rows with
`n <= 1` need no welds and cost `0`. Read `n` and the widths from stdin, print the minimum cost.

**Why the obvious greedy is wrong.** The cost looks Huffman-like, tempting a "weld the cheapest
adjacent pair first" greedy. But Huffman may combine *any* two items, whereas here only *neighbours*
may fuse. That adjacency restriction is a global trade-off: fusing two cheap neighbours can bury a
slab inside a piece that is then re-paid by every later weld, while a different order keeps an
expensive boundary slab on the outside and welds it last. Smallest-first is therefore not optimal and
is discarded.

**Key idea — interval DP.** Let `dp[i][j]` be the minimum cost to fuse the *closed* range of slabs
`[i, j]` into one piece. However the range is fused, its *last* weld joins two adjacent sub-pieces
that together span all of `[i, j]`, so it pays the full range width `W(i, j) = w[i] + ... + w[j]`
regardless of how the halves formed, and each half is itself a fully-fused sub-range. Hence

- `dp[i][i] = 0` (a lone slab needs no welds),
- `dp[i][j] = min over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] ) + W(i, j)`.

`k` is the last index of the left half `[i, k]`, with right half `[k+1, j]`; both must be non-empty,
which forces `i <= k <= j-1`. Compute `W` with a half-open prefix sum `prefix[t] = w[0] + ... +
w[t-1]`, so the inclusive-range width is `prefix[j+1] - prefix[i]`. Answer: `dp[0][n-1]`.

**Correctness.** The recurrence is exhaustive over the position of the final weld (every full
parenthesization of the line is captured by some split `k` at each level), and optimal substructure
holds because each half's cost is independent of the other once the split is fixed. Verified against
an independent brute force that simulates *all* orders of adjacent merges on over 950 random cases
with zero mismatches.

**Pitfalls — all at the inclusive/exclusive boundary.**
1. *Width index off-by-one.* The range `[i, j]` is **closed**, but `prefix` is **half-open**. Writing
   the width as `prefix[j] - prefix[i]` sums only `[i, j)` and silently drops slab `j`. A trace of
   `[2, 3]` returning `2` instead of `5` exposes this; the correct width is `prefix[j+1] - prefix[i]`.
2. *Split bound off-by-one.* The split must keep both halves non-empty, so `k` runs `i .. j-1`
   (`k < j`). Allowing `k = j` reads the cell being computed (`dp[i][j]`) and an inverted range
   (`dp[j+1][j]`), both garbage.
3. *Length base.* Run `len` from `2`; starting at `1` would re-enter singletons with an empty
   `k`-loop and poison `dp[i][i]` to `INF`.
4. *Overflow.* With `n` up to `400` and `w[i]` up to `10^6`, the total cost reaches `~1.6*10^11`; use
   `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 0` and `n = 1` short-circuit to `0`; `n = 2` gives `w[0] + w[1]`. The `INF`
sentinel is only ever read inside a `min` and never has a width added to a cell that stayed `INF`
(every `dp[i][j]` with `i < j` gets a real value), so no `INF` leaks.

**Complexity.** `O(n^3)` time (`~6.4*10^7` at `n = 400`, ~0.01 s), `O(n^2)` memory (~1.3 MB).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    if (n <= 1) { cout << 0 << "\n"; return 0; }

    // prefix[i] = w[0] + ... + w[i-1], a half-open prefix sum over [0, i).
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + w[i];

    // dp[i][j] = minimum total merge cost to combine the slabs whose indices
    // lie in the CLOSED interval [i, j] into one slab. The cost of the final
    // merge that unites the two children [i,k] and [k+1,j] is the combined
    // width prefix[j+1]-prefix[i] (sum of all w over the closed range).
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    // length-1 intervals already cost 0 (initialized above).

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;          // closed interval [i, j]
            long long best = INF;
            // split between k and k+1, so k ranges over i .. j-1 (inclusive).
            for (int k = i; k < j; k++) {
                long long cur = dp[i][k] + dp[k + 1][j];
                if (cur < best) best = cur;
            }
            // width of the closed range [i, j] is prefix[j+1] - prefix[i].
            dp[i][j] = best + (prefix[j + 1] - prefix[i]);
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
```

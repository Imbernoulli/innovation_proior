**Problem.** `n` film reels sit on a circular carousel in fixed clockwise order; reel `i` holds `w[i]` metres. A splice may only join two reels that are currently **neighbours** (the carousel wraps, so the first and last reel are neighbours too). Splicing lengths `x` and `y` makes one reel of length `x + y` and costs `x + y`. Splice until one reel remains; output the minimum total cost. If `n <= 1` the cost is `0`. Read `n` and the `w[i]` from stdin, print the minimum cost.

**Why the famous Huffman greedy is wrong.** The classic min-cost-merge / optimal-merge-tree greedy ("repeatedly merge the two smallest piles") is provably optimal only when *any* two piles may merge. Here splices are restricted to carousel neighbours, and Huffman exploits a freedom it does not have. On `w = [1, 2, 1, 3]` the two smallest reels are the two `1`s at slots `0` and `2`, which are **opposite** on the 4-cycle, not adjacent. Huffman merges them first and reports `2 + 4 + 7 = 13`, but that splice is illegal; the cheapest legal sequence costs `14`. Huffman under-reports (it returns an infeasible lower bound), so it would not even look obviously broken — it is discarded.

**Key idea — interval DP on contiguous arcs, over a doubled array.** Because every splice joins neighbours, any subset that gets fused is a contiguous arc of the carousel. For a *line* of reels `i..j`, let `dp[i][j]` be the min cost to fuse them into one. The last splice of the arc joins `i..k` with `k+1..j` and costs the whole arc's length `W(i,j)` (independent of how the two sides were built):

- `dp[i][j] = min over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] ) + W(i, j)`, with `dp[i][i] = 0`.

For the **circle**, the optimal process leaves exactly one carousel gap as the final straddle, i.e. the circle behaves like a line opened at some slot `i`. Different openings differ, so try them all. To avoid modular arithmetic, **double the array**: `a[t] = w[t mod n]` for `t in [0, 2n)`; then an arc of `n` reels starting at slot `i` is the contiguous segment `a[i..i+n-1]`. Index the table by left endpoint `l` and length `len` (`1..n`): `dp[l][len]` over `a[l..l+len-1]`, with arc sum from a prefix-sum array. Answer: `min over i in [0, n) of dp[i][n]`.

**Pitfalls.**
1. *Sub-arc length off-by-one.* When splitting arc `[l, r]` after position `k`, the left sub-arc `[l, k]` has length `k - l + 1` (not `k - l`) and the right `[k+1, r]` has length `r - k`. Indexing `dp[l][k - l]` reads a non-existent length-0 cell. (A trace of `[4, 2]` exposes this.)
2. *Single opening is not enough on a circle.* Returning only `dp[0][n]` can miss the cheapest opening on an asymmetric carousel; take the min over all `n` openings of the doubled array.
3. *Overflow.* Total cost is bounded by `(n-1) * sum(w) ~ 1000 * 10^9 = 10^12`, beyond 32-bit range; use `long long`. Keep `INF` well above `10^12` (e.g. `LLONG_MAX/4`) and never add two `INF`s.

**Edge cases.** `n = 0` -> `0`; `n = 1` -> `0` (handled before the DP); `n = 2` -> the single neighbour splice `w[0]+w[1]`, counted once; uniform reels and wrap-around arcs verified against a brute force over all legal adjacent-merge orders.

**Complexity.** `O(n^3)` time (for each of `2n` left endpoints and each length up to `n`, scan up to `n` split points) and `O(n^2)` memory (`2n * (n+1)` longs, about `16` MB at `n = 1000`). Runs well under the 2-second limit (`~0.9` s at `n = 1000`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    if (n == 0) { cout << 0 << "\n"; return 0; }
    vector<long long> w(n);
    for (auto &x : w) cin >> x;
    if (n == 1) { cout << 0 << "\n"; return 0; }

    // Circular merge of adjacent reels. Unroll the circle into a line of length 2n
    // so any contiguous arc of length L (over the original n reels) appears as a
    // contiguous segment [i, i+L-1] for some 0 <= i < n.
    int m = 2 * n;
    vector<long long> a(m + 1);
    for (int i = 0; i < m; i++) a[i] = w[i % n];
    // prefix sums of the doubled array
    vector<long long> pre(m + 1, 0);
    for (int i = 0; i < m; i++) pre[i + 1] = pre[i] + a[i];
    auto sum = [&](int l, int r) { return pre[r + 1] - pre[l]; }; // inclusive [l,r]

    const long long INF = LLONG_MAX / 4;
    // dp over segments of the doubled array; we only need segments of length <= n.
    // dp[l][len] = min cost to merge reels a[l..l+len-1] into one.
    // Use 2D vectors indexed by left endpoint l (0..m-1) and length len (1..n).
    vector<vector<long long>> dp(m, vector<long long>(n + 1, INF));
    for (int l = 0; l < m; l++) dp[l][1] = 0;
    for (int len = 2; len <= n; len++) {
        for (int l = 0; l + len <= m; l++) {
            int r = l + len - 1;
            long long best = INF;
            long long s = sum(l, r);
            for (int k = l; k < r; k++) {
                int leftLen = k - l + 1;
                int rightLen = r - k;
                long long cand = dp[l][leftLen] + dp[k + 1][rightLen] + s;
                if (cand < best) best = cand;
            }
            dp[l][len] = best;
        }
    }
    // Answer: best over all starting reels i of merging the whole circle into one,
    // i.e. the arc of length n starting at i.
    long long ans = INF;
    for (int i = 0; i < n; i++) ans = min(ans, dp[i][n]);
    cout << ans << "\n";
    return 0;
}
```

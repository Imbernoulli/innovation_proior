**Problem.** Given `n` keys in fixed sorted order with non-negative access frequencies `f[1..n]`, store them in a binary search tree. The cost of a tree is `sum over i of depth(i) * f[i]` (root at depth 1), and the in-order traversal must list the keys in sorted order. Output the minimum total cost over all BST shapes; for `n = 0` the cost is `0`. Read `n` and the frequencies from stdin, print one integer.

**Why the obvious greedy is wrong.** "Make the most frequently accessed key the root, then recurse on the two halves" fails because the BST search-order constraint couples every key's depth to the root choice — the root's position fixes the entire left/right partition. On `f = [2, 3, 4]` greedy roots at key 3 (frequency 4) and is forced to chain keys 2 and 1 below it: cost `4*1 + 3*2 + 2*3 = 16`. But rooting at the *middle* key 2 keeps both other keys at depth 2: cost `3*1 + 2*2 + 4*2 = 15 < 16`. A local "biggest at the root" rule cannot weigh the global partition, so greedy is discarded.

**Key idea — interval DP over contiguous key ranges.** Because the keys are sorted, choosing key `r` as the root of a range `[i..j]` forces keys `i..r-1` into the left subtree and `r+1..j` into the right. Let `dp[i][j]` be the minimum cost of an optimal BST on keys `i..j`, with that subtree's own root counted at depth 1; the empty range has `dp[i][i-1] = 0`. When two optimal subtrees are hung under a new root, every key inside them sinks exactly one level, adding the subtree's total weight; combined with the root's own frequency this is just the whole range's weight `W(i,j) = f[i] + ... + f[j]` charged once:

- `dp[i][j] = W(i, j) + min over root r in [i..j] of ( dp[i][r-1] + dp[r+1][j] )`

The `W(i,j)` term, added once per level of recursion, is exactly what makes each key pay `depth * f`. Answer: `dp[1][n]`. Precompute prefix sums so `W(i,j) = prefix[j] - prefix[i-1]` is `O(1)`, and fill `dp` by increasing range length.

**Two pitfalls to get right.**
1. *The one-level-sink weight.* Forgetting the `+ W(i, j)` term (treating `dp` as a plain sum of subtree costs) charges nothing for descending a level and collapses every answer toward `0`. A trace of the singleton `f = [5]` returning `0` instead of `5` exposes exactly this. Add the weight exactly once — `dp[i][j] = best + w` — not inside each candidate *and* again on assignment (which double-counts, returning `10` for `[5]`).
2. *Overflow.* With `n` up to `500` and `f[i]` up to `10^9`, a key at depth up to `500` makes the cost reach `~2.5 * 10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by the recurrence + zero-initialized table):** `n = 0` -> `0` (empty tree); `n = 1` -> `f[1]`; all-zero frequencies -> `0`; all-equal frequencies -> the balanced optimum (not the greedy chain); a single huge frequency -> the DP reaches the same near-root placement greedy would, by minimization.

**Complexity.** `O(n^2)` ranges times an `O(n)` root scan is `O(n^3) ≈ 2 * 10^7` for `n = 500`, which runs in well under 0.1s — comfortably inside the 2-second limit, so the simple provable cubic DP is shipped rather than the fiddlier Knuth–Yao `O(n^2)` optimization. `O(n^2)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // n = 0 (or empty input) -> cost 0
    vector<long long> f(n + 1);
    for (int i = 1; i <= n; i++) cin >> f[i];

    // prefix[i] = f[1] + ... + f[i]; weight of interval [i..j] is prefix[j]-prefix[i-1].
    vector<long long> prefix(n + 1, 0);
    for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + f[i];

    // dp[i][j] = minimum expected cost (sum over depths*freq, root at depth 1)
    //            for an optimal BST built from keys i..j (1-indexed, inclusive).
    // dp[i][i-1] = 0 represents an empty range.
    // Recurrence: dp[i][j] = (prefix[j]-prefix[i-1])
    //                        + min over root r in [i..j] of dp[i][r-1] + dp[r+1][j].
    // The added interval weight accounts for every key in [i..j] sinking one level
    // deeper when we hang the two subtrees under the chosen root.
    vector<vector<long long>> dp(n + 2, vector<long long>(n + 2, 0));

    // len = number of keys in the interval, from 1 up to n.
    for (int len = 1; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            long long w = prefix[j] - prefix[i - 1];
            long long best = LLONG_MAX;
            for (int r = i; r <= j; r++) {
                long long left = dp[i][r - 1];      // r==i => empty left, dp[i][i-1]=0
                long long right = dp[r + 1][j];     // r==j => empty right, dp[j+1][j]=0
                long long cand = left + right;
                if (cand < best) best = cand;
            }
            dp[i][j] = best + w;
        }
    }

    cout << dp[1][n] << "\n";                // dp[1][0] = 0 when n == 0
    return 0;
}
```

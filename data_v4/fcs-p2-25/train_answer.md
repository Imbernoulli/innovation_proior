**Problem.** Given an `n x n` grid `a[i][j]` (values may be negative), a falling path starts at any cell in the top row and from `(i, j)` drops to one of `(i+1, j-1)`, `(i+1, j)`, `(i+1, j+1)` inside the grid, ending at the bottom row. Its cost is the sum of its `n` cells. Read `n` and the row-major grid from stdin; print the minimum falling-path cost. For `n = 0`, print `0`.

**Why the obvious greedy is wrong.** "Start at the cheapest top cell, then always step to the cheapest of the three reachable cells below" fails because a falling path is a global object: the column you occupy at row `i` constrains which columns you can reach below. On

```
5 5 5
1 9 9
50 50 1
```

greedy (starting anywhere in the uniform top row, say column 0) takes the cheap `1` at row 1, which pins it to column 0, whose only descendants at row 2 are the two `50`s: cost `5 + 1 + 50 = 56`. The optimal path takes the *worse-looking* `9` at row 1 to keep column 2 reachable, landing on the lone `1` at row 2: columns `0 -> 1 -> 2`, cost `5 + 9 + 1 = 15`. A cheap step early buys an expensive fence. Greedy is discarded.

**Key idea — layered DP (shortest path on a DAG, row by row).** Let `dp[i][j]` be the minimum cost of a falling path that *ends* at cell `(i, j)`. The future of a partial path depends only on its current cell, so this has clean optimal substructure. Base case is the top row, and each cell relaxes from its (at most) three predecessors directly above:

- `dp[0][j] = a[0][j]`  (a one-cell path)
- `dp[i][j] = a[i][j] + min( dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1] )`  (skip out-of-range predecessors)

The answer is `min over j of dp[n-1][j]`. Every `dp` value equals the cost of a real path, and the `min` ranges over every legal predecessor, so the DP is provably optimal — unlike greedy. Process rows in order, keeping just the previous row; `O(n^2)` time, `O(n)` extra space.

**Two pitfalls to get right.**
1. *Edge columns.* Columns `0` and `n-1` have only two predecessors. Guard the reads — start `best = dp[j]` (the always-valid straight-down predecessor), fold in `dp[j-1]` only when `j > 0` and `dp[j+1]` only when `j+1 < n`. No sentinels, so no fabricated value can leak into a `min`, and no index escapes `[0, n-1]`.
2. *In-place row corruption.* Overwriting `dp[j]` while computing row `i` shreds the previous row: a later cell's predecessor read picks up an already-updated same-row value, which is not a legal predecessor. (A trace of `[[1,100],[100,1]]` returning `101` instead of `2` exposes exactly this.) Compute each row into a fresh vector from the untouched previous row, then swap.

**Other corners.** `n = 0` -> print `0` up front. `n = 1` -> the single cell is the only path, so a negative single cell is the (optimal) answer; there is **no** "empty allowed" floor, because a falling path must cross every row — do not clamp at `0`. Overflow: a path sums up to `n * 10^9 = 10^12` in magnitude, past 32-bit range, so use `long long`.

**Complexity.** `O(n^2)` time, `O(n)` extra space. At `n = 1000` this runs in well under a tenth of a second.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no input -> nothing to do
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<vector<long long>> a(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> a[i][j];

    // dp[j] = minimum sum of a falling path that ends at cell (current row, j).
    // Row 0: a path ending at (0, j) is just the single cell a[0][j].
    vector<long long> dp(a[0].begin(), a[0].end());

    for (int i = 1; i < n; i++) {
        vector<long long> ndp(n);
        for (int j = 0; j < n; j++) {
            long long best = dp[j];                       // came from (i-1, j)
            if (j > 0)     best = min(best, dp[j - 1]);    // came from (i-1, j-1)
            if (j + 1 < n) best = min(best, dp[j + 1]);    // came from (i-1, j+1)
            ndp[j] = best + a[i][j];
        }
        dp = move(ndp);
    }

    long long answer = *min_element(dp.begin(), dp.end());
    cout << answer << "\n";
    return 0;
}
```

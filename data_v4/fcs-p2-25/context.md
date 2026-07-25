# Minimum falling path sum through an n x n grid

## Research question

You are given an `n x n` grid `a[0..n-1][0..n-1]` of integers (values may be negative). A
**falling path** starts at any cell in the top row (`row 0`) and, from a cell at `(i, j)`, moves
to one of the (at most) three cells directly below it in the next row: `(i+1, j-1)`, `(i+1, j)`,
or `(i+1, j+1)`, never stepping outside the grid. A path ends when it reaches the bottom row.

The cost of a path is the sum of the values of the cells it visits (one cell per row, so a path
through an `n x n` grid visits exactly `n` cells). Output the **minimum** possible cost over all
falling paths.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 1000`). Then follow `n*n` integers, the grid
  in row-major order (`a[0][0] a[0][1] ... a[0][n-1] a[1][0] ...`), whitespace-separated. Each
  value satisfies `-10^9 <= a[i][j] <= 10^9`.
- Output (stdout): a single line with the minimum falling-path cost. For `n = 0` (empty grid, no
  path), output `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the grid

```
2 1 3
6 5 4
7 8 9
```

the answer is `13` (path `1 -> 4 -> 8`, columns `1 -> 2 -> 1`).

## Evaluation settings

Judged on hidden tests covering: all-positive grids; grids with negatives and zeros; the empty grid
(`n = 0`); a single cell (`n = 1`, including a single negative); adversarial grids designed to
penalize short-sighted step-by-step choices; uniform grids; and large `n = 1000` with values near
`10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<vector<long long>> a(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> a[i][j];

    // TODO: compute the minimum-cost falling path from the top row to the bottom row,
    // where from (i, j) a path may move to (i+1, j-1), (i+1, j), or (i+1, j+1).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

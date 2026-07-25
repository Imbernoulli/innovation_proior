# Counting pairs whose absolute difference falls in a band

## Research question

You are given an array `a[0..n-1]` of integers and two integers `L` and `R` with `0 <= L <= R`.
Count the number of **unordered** index pairs `{i, j}` with `i != j` such that

```
L <= |a[i] - a[j]| <= R.
```

Each unordered pair must be counted **exactly once** (so `{i, j}` and `{j, i}` are the same pair).
Output that count.

This is the standard "how many pairs land in a value band" question that appears inside near-duplicate
detection, histogram bucketing, and offline range-counting problems.

## Input / output contract

- Input (stdin): the first line holds three integers `n`, `L`, `R`
  (`0 <= n <= 2*10^5`, `0 <= L <= R <= 2*10^9`). If `n > 0`, the second line holds the `n` integers
  `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated. If `n = 0` there are no further tokens.
- Output (stdout): a single line with the number of qualifying unordered pairs.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `L = 2`, `R = 4`, `a = [1, 5, 3, 8, 6, 2]` the answer is `8`.

## Background

The brute force is `O(n^2)`: test every pair. With `n` up to `2*10^5` that is `~2*10^10` pair tests,
far too slow, so an offline counting structure is needed. Sorting-based sliding windows and Fenwick
(BIT) structures over compressed values are both standard tools for this kind of range-counting
problem.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` (answer `0`); arrays with many equal values; the
`L = 0` band (including `L = R = 0`, which counts equal-value pairs); one-sided extreme bands where no
pair qualifies; values spanning the full `[-10^9, 10^9]` range with `R` up to `2*10^9`; and large
`n = 2*10^5` so an `O(n^2)` brute force times out.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count unordered pairs {i,j} with L <= |a[i]-a[j]| <= R.
    long long ans = 0;

    cout << ans << "\n";
    return 0;
}
```

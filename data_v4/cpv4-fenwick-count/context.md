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
detection, histogram bucketing, and offline range-counting problems. The 1-D version looks innocent, but
the band is two-sided around each element, and the `L = 0` corner makes the two sides touch — which is
exactly where a Fenwick-based count silently double-counts.

## Input / output contract

- Input (stdin): the first line holds three integers `n`, `L`, `R`
  (`0 <= n <= 2*10^5`, `0 <= L <= R <= 2*10^9`). If `n > 0`, the second line holds the `n` integers
  `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated. If `n = 0` there are no further tokens.
- Output (stdout): a single line with the number of qualifying unordered pairs.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `L = 2`, `R = 4`, `a = [1, 5, 3, 8, 6, 2]` the answer is `8`.

## Background

The brute force is `O(n^2)`: test every pair. With `n` up to `2*10^5` that is `~2*10^10` pair tests,
far too slow, so an offline counting structure is needed. Two ideas are on the table before committing:

- **Sort + two pointers.** Sort the values; for a fixed band `[L, R]` the set of partners of each value
  is a contiguous slice of the sorted array, which a sliding window can count. This is `O(n log n)`. The
  open questions are how to handle the two-sided band (partners both below and above) and how to avoid
  counting a pair from both endpoints.
- **Fenwick (BIT) over compressed values.** Sweep the array once; for each new element `a[j]` query, in
  `O(log n)`, how many already-seen elements have a value inside the band, then insert `a[j]`. Processing
  `j` left to right and querying only earlier elements makes each unordered pair counted from its later
  endpoint exactly once. The open questions are the inclusive-range query indices after coordinate
  compression, and what happens to the two value bands when `L = 0`.

The count can be as large as `C(n, 2) = ~2*10^10`, which exceeds 32 bits, so the accumulator must be
64-bit.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` (answer `0`); arrays with many equal values; the
`L = 0` band (including `L = R = 0`, which counts equal-value pairs); one-sided extreme bands where no
pair qualifies; values spanning the full `[-10^9, 10^9]` range with `R` up to `2*10^9`; and large
`n = 2*10^5` so an `O(n^2)` brute force times out and a 32-bit accumulator overflows.

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

    // TODO: count unordered pairs {i,j} with L <= |a[i]-a[j]| <= R, using a
    // Fenwick tree over compressed values; query earlier elements per j.
    long long ans = 0;

    cout << ans << "\n";
    return 0;
}
```

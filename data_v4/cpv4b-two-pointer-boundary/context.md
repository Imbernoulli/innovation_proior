# Counting stable stretches in glacier ice-thickness readings

## Research question

A glacier monitoring station records one ice-thickness reading per day, giving a sequence of `n`
integers `a[0..n-1]` (a reading can be negative when measured as a deviation from a baseline). The
analysts call a contiguous run of days a **stretch**, and they call a stretch **stable** when the
difference between the thickest and the thinnest reading inside it is **strictly less than** a
tolerance `D` millimetres.

Count how many non-empty contiguous stretches (subarrays) are stable. Formally, count the pairs
`(i, j)` with `0 <= i <= j < n` such that
`max(a[i..j]) - min(a[i..j]) < D`. Output that count.

The single-element stretch `[i, i]` has `max - min = 0`, so it is stable exactly when `D > 0`.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `D` (`0 <= n <= 2*10^5`,
  `0 <= D <= 2*10^9`). The second line (present only when `n > 0`) holds the `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the number of stable stretches.
- Time limit: 1 second. Memory: 256 MB.

Example: for `D = 5` and `a = [4, 8, 6, 11, 9, 7]` the answer is `12`.

## Evaluation settings

Judged on hidden tests covering: `D = 0`; all-equal arrays with `D = 1`; arrays whose window span
lands exactly on `D`; the empty array (`n = 0`); a single element with `D = 0` and with `D = 1`;
large `n = 2*10^5` with values near `+-10^9`; and mixed sign / negative readings.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count contiguous subarrays whose (max - min) < D.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

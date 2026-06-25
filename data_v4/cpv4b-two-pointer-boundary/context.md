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

The single-element stretch `[i, i]` has `max - min = 0`, so it is stable exactly when `D > 0`. The
whole point is the boundary: whether `< D` or `<= D`, and whether a window-ending count adds
`right - left + 1` or `right - left`, decides correctness, and the failure only shows up on small
traced cases (a one-element window with `D = 0`, or a window whose span lands exactly on `D`).

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `D` (`0 <= n <= 2*10^5`,
  `0 <= D <= 2*10^9`). The second line (present only when `n > 0`) holds the `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the number of stable stretches.
- The answer can be as large as `n*(n+1)/2 = 2*10^10`, which exceeds 32-bit range, so it must be
  printed from a 64-bit accumulator.
- Time limit: 1 second. Memory: 256 MB.

Example: for `D = 5` and `a = [4, 8, 6, 11, 9, 7]` the answer is `12`.

## Background

This is a "count subarrays satisfying a monotone window predicate" problem. The predicate
`max - min < D` is **monotone in the window**: if a window is stable, every sub-window of it is also
stable (shrinking a window can only shrink its range). That monotonicity is exactly what a
two-pointer sweep exploits. Two families of approach are on the table before committing:

- **Quadratic enumeration.** For every start `i`, extend the end `j` and maintain the running max and
  min, counting while the range stays below `D`. This is `O(n^2)`, obviously correct, and a perfect
  brute-force oracle, but too slow for `n = 2*10^5`.
- **Sliding window with monotonic deques.** Move a right pointer across the array; for each `right`,
  advance a left pointer to the smallest start that keeps the window stable, and add the number of
  valid starts. Maintaining the window max and min in amortized `O(1)` needs two monotonic deques.
  This is `O(n)`; the open question is the exact left-advance condition and the exact count
  increment — the inclusive/exclusive boundary.

## Evaluation settings

Judged on hidden tests covering: `D = 0` (no stretch is ever stable, answer `0`); all-equal arrays
with `D = 1` (every stretch stable, answer `n*(n+1)/2`); arrays whose window span lands exactly on
`D` (the strict-vs-non-strict boundary); the empty array (`n = 0`); a single element with `D = 0`
and with `D = 1`; large `n = 2*10^5` with values near `+-10^9` so the answer overflows 32 bits; and
mixed sign / negative readings.

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

    // TODO: count contiguous subarrays whose (max - min) < D, in O(n)
    //       using a sliding window with two monotonic deques.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

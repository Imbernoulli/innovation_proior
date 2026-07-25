# Counting subarrays with strictly negative sum

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be **negative, zero, or
positive**). Count the number of contiguous subarrays whose sum is **strictly less than zero**.
A subarray is identified by a pair `(l, r)` with `0 <= l <= r < n`, and its sum is
`a[l] + a[l+1] + ... + a[r]`. Output how many such pairs have `sum(l, r) < 0`.

This is the "count subarrays whose sum lands in a range" pattern reduced to its sharpest corner:
the range is the open half-line `(-inf, 0)`. It shows up inside balance-tracking, risk-window,
and signal-threshold problems, so getting the one-dimensional version exactly right — including
the all-negative, all-zero, and empty-array corners and the strict-vs-nonstrict boundary at `0` —
matters.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` there are no further tokens.
- Output (stdout): a single line with the count of subarrays whose sum is strictly negative.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, -4, 1, -2]` the answer is `7`. The qualifying subarrays (by index range)
are `[0,1]=-1`, `[0,3]=-2`, `[1,1]=-4`, `[1,2]=-3`, `[1,3]=-5`, `[2,3]=-1`, `[3,3]=-2`.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, the empty
array (`n = 0`), a single element (negative, zero, positive), all-negative arrays, all-zero arrays,
and large `n = 2*10^5` with values near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count subarrays (l, r) with sum(a[l..r]) < 0
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

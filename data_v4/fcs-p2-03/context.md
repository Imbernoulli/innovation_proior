# Maximum product of a contiguous subarray

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be negative, zero, or
positive). Among all **contiguous, non-empty** subarrays `a[i..j]` (`0 <= i <= j < n`),
find the one whose **product** of elements is the largest, and output that maximum product.

Because the subarray must be non-empty, a single element is always a valid subarray, so the
answer is simply the largest element when every longer window is worse (for example on an
all-zero or strictly-decreasing-magnitude input).

This is the multiplicative analogue of the classic maximum-subarray (Kadane) problem, and it
is a standard building block: it appears when scoring multiplicative gains/losses over a
window, compounding ratios, or sign-sensitive streak detection.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 18`); then `n` integers `a[i]`
  (`-9 <= a[i] <= 9`), whitespace-separated (any mix of spaces and newlines).
- Output (stdout): a single line with the maximum achievable product over all non-empty
  contiguous subarrays.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [2, 3, -2, 4]` the answer is `6` (the subarray `[2, 3]`). The window
`[2, 3, -2, 4]` has product `-48` and `[-2, 4]` has product `-8`, so neither beats `6`.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays with negatives and zeros,
single element (`n = 1`) including a single negative, all-negative arrays, arrays containing
zeros, and maximal `n = 18` with values near `±9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum product over all non-empty contiguous subarrays.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

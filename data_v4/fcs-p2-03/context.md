# Maximum product of a contiguous subarray

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be negative, zero, or
positive). Among all **contiguous, non-empty** subarrays `a[i..j]` (`0 <= i <= j < n`),
find the one whose **product** of elements is the largest, and output that maximum product.

Because the subarray must be non-empty, a single element is always a valid subarray, so the
answer is simply the largest element when every longer window is worse (for example on an
all-zero or strictly-decreasing-magnitude input). Sign interacts with the objective in a way
that pure max-sum problems never see: multiplying by a negative value can turn the *smallest*
running product into the *largest*, so the structure of the optimum is genuinely different
from the additive case.

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

## Constraints and magnitudes

With `n <= 18` and `|a[i]| <= 9`, the magnitude of any contiguous product is at most
`9^18 = 150094635296999121 ≈ 1.5 * 10^17`, which fits comfortably inside a signed 64-bit
integer (`long long`, max `≈ 9.2 * 10^18`). So exact integer arithmetic is available end to
end; no overflow handling, no floating point, no big integers are required as long as 64-bit
accumulators are used. Using a 32-bit `int` would overflow on long high-magnitude windows and
is a silent wrong answer.

## Background

Two routes are on the table before committing to one:

- **Kadane-on-product.** Adapt the additive maximum-subarray scan directly: keep a single
  running "best product ending here", `cur = max(a[i], cur * a[i])`, and track the global
  maximum. It is `O(n)` and a three-line transcription of the well-known sum version. The open
  question is whether carrying only the running maximum is sound once factors can be negative.
- **Min/max product DP.** Scan left to right carrying, for each prefix, *both* the maximum and
  the minimum product of a subarray ending at the current position. This is `O(n)`, `O(1)`
  memory. The open question is the exact set of transition candidates and why the minimum has
  to be carried alongside the maximum.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays with negatives and zeros,
single element (`n = 1`) including a single negative, all-negative arrays (where an even-length
window flips to a large positive), arrays where a `0` resets the running window, and maximal
`n = 18` with values near `±9` (so the running product is large but still inside 64 bits).

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

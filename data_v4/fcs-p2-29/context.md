# Maximum subarray sum with at most one element deleted

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be negative). Consider every
**contiguous, non-empty** subarray `a[l..r]`. From the chosen subarray you may delete **at most one**
element (deleting zero is allowed), with the rule that what remains must still be **non-empty** — so a
length-`1` subarray cannot have its single element removed. The score of a subarray is the sum of the
elements that remain. Output the maximum score over all subarrays and all allowed deletions.

Because every subarray of length `1` is allowed (with no deletion), the answer is always defined even
when all values are negative; in that case the best you can do is the single least-negative element.

This is the "delete at most one" variant of the classic maximum-subarray problem. The deletion option
shows up whenever a window is allowed to discard one outlier (a single bad reading, one penalty cell),
so getting the negative-value and non-empty corners exactly right matters.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable score.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [5, 6, -3, 3, -5]` the answer is `14` — take the subarray `[5, 6, -3, 3]` and delete
the `-3`, leaving `5 + 6 + 3 = 14`.

## Background

The "at most one deletion" twist makes a purely local rule risky, because the best window to delete
from need not be the best window without a deletion. Two families of approach are on the table before
committing to one:

- **Plain max-subarray then delete the worst element.** Run ordinary Kadane to find the best subarray
  with no deletion, then, inside that one window, drop its most negative element if that helps. This is
  `O(n)` and a couple of lines on top of Kadane; the open question is whether the optimal window to
  delete from is really the same window Kadane already found.
- **Two-state Kadane DP.** Scan left to right maintaining, for each prefix ending at `i`, the best
  subarray-ending-here sum with **no** deletion used and with **exactly one** deletion used. This is
  `O(n)`; the open question is the exact recurrence and the order in which the two states reference each
  other.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, single element
(`n = 1`), all-negative arrays (answer is the largest single element), windows where the profitable
deletion extends the window past where plain Kadane would stop, and large `n = 10^5` with values near
`10^9` (so a running sum can exceed a 32-bit integer).

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
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum sum over all non-empty contiguous subarrays
    //       when at most one element may be deleted (remainder must stay non-empty).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

# Counting pairs whose sum is at most a threshold

## Research question

You are given `n` integers `a[0..n-1]` (values may be negative, zero, or positive) and an integer
threshold `T`. Count the number of **unordered pairs** of distinct positions `(i, j)` with `i < j`
such that `a[i] + a[j] <= T`. Output that count.

This is the "count-pairs-under-a-bound" primitive that sits inside two-sum variants, scheduling
feasibility checks, and the merge step of divide-and-conquer counting. The one-dimensional version
has to be exactly right — including the all-negative array, the `n < 2` corner, a threshold `T` that
is itself negative, and the fact that pair sums can overflow a 32-bit integer.

## Input / output contract

- Input (stdin): the first line has two integers `n` (`0 <= n <= 2*10^5`) and `T`
  (`-2*10^9 <= T <= 2*10^9`); then `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`),
  whitespace-separated (they may span one or several lines).
- Output (stdout): a single line with the count of pairs `(i, j)`, `i < j`, with `a[i] + a[j] <= T`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `T = 3`, `a = [-2, 0, 5, 1, -3, 4]` the answer is `10`. (Sorted the values are
`[-3, -2, 0, 1, 4, 5]`; ten of the fifteen pairs have sum `<= 3`.)

## Background

The count is over all `O(n^2)` pairs, but `n` is large, so the quadratic brute force is only a
reference, not a solution. Two families of approach are on the table before committing to one:

- **Sort + binary search.** Sort the values; for each index `i`, the partners `j > i` with
  `a[j] <= T - a[i]` form a prefix of the sorted suffix, found by one `upper_bound`. This is
  `O(n log n)`; the open question is getting the "distinct indices, count each pair once" bookkeeping
  right so a value is never paired with itself or a pair double-counted.
- **Sort + two-pointer sweep.** Sort the values and sweep a low pointer up and a high pointer down. At
  each step, if `a[lo] + a[hi] <= T`, then because the array is sorted *every* index in `[lo, hi-1]`
  also pairs with `hi`, so add `hi - lo` at once and advance `lo`; otherwise lower `hi`. This is
  `O(n)` after the sort and avoids per-element searches; the open question is the exact loop
  condition and what to add at each step.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, the empty
array (`n = 0`), a single element (`n = 1`, no pair exists), all-negative arrays with a negative `T`,
threshold exactly equal to a pair sum (the `<=` boundary), heavy duplicate values, and large
`n = 2*10^5` with values near `+/-10^9` (so a pair sum reaches `2*10^9` and the *count* reaches
`~2*10^10`, both beyond 32-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count unordered pairs (i < j) with a[i] + a[j] <= T.
    long long ans = 0;

    cout << ans << "\n";
    return 0;
}
```

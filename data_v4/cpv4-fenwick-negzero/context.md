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

## Background

Write `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]` for `1 <= k <= n` (the prefix-sum array of
length `n + 1`). The sum of subarray `(l, r)` equals `P[r+1] - P[l]`. So
`sum(l, r) < 0  <=>  P[r+1] < P[l]`, and the answer is the number of pairs `(i, j)` with
`0 <= i < j <= n` and `P[j] < P[i]` — the count of *strict inversions* of the prefix array.

Two families of approach are on the table before committing:

- **Brute force over subarrays.** For every `l`, accumulate the running sum to the right and
  tally each negative prefix. `O(n^2)`, trivial to write, correct by construction — but `n` up to
  `2*10^5` makes `~2*10^10` operations, far over the time limit. Useful only as an oracle.
- **Fenwick over compressed prefix sums.** Sweep `j = 0..n` in index order; before inserting
  `P[j]`, query how many already-inserted `P[i]` (with `i < j`) are strictly greater than `P[j]`.
  A Binary Indexed Tree over the coordinate-compressed prefix-sum values answers each query in
  `O(log n)`, for `O(n log n)` overall. The open questions are the exact "count greater" query
  (Fenwicks naturally answer "count `<= x`"), the strict boundary, and how the compression and
  base index `P[0] = 0` interact with negative and zero values.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays (answer `0`), arrays mixing negatives and
zeros, the empty array (`n = 0`), a single element (negative, zero, positive), all-negative arrays,
all-zero arrays (answer `0`, since `0` is not strictly negative), and large `n = 2*10^5` with
values near `10^9` so the count can exceed a 32-bit integer.

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

    // TODO: count subarrays (l, r) with sum(a[l..r]) < 0, using a Fenwick tree
    //       over the coordinate-compressed prefix sums P[0..n] (P[0] = 0).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

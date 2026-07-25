# Exactly-k Disjoint Segments — maximum total job value

## Research question

You are given an integer array `a[0..n-1]` (values may be negative). A *job* is a
non-empty contiguous segment `a[l..r]`, and its value is the sum of the elements it
covers. You must choose **exactly `k`** jobs that are pairwise disjoint (no two share a
cell) and **maximize the total value** of the chosen jobs. Output that maximum total.

This is the "pick exactly `k` disjoint maximum-sum subarrays" problem. The "exactly `k`"
constraint is what makes it hard: you are not free to take only the profitable segments
and stop — if `k` is large you may be *forced* to open extra segments that drag the total
down, and if `k` is small you must leave value on the table. The selection is global, so
the right number of segments at each position depends on the whole array.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k`
  (`1 <= n <= 2*10^5`, `1 <= k <= ceil(n/2)`); the second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable total value over all ways to
  choose exactly `k` disjoint non-empty contiguous segments.
- The bound `k <= ceil(n/2)` guarantees a valid selection always exists (`k` non-empty
  segments separated by at least one gap need at least `2k-1` cells).
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8`, `k = 2`, `a = [3, -1, 4, -1, 5, -9, 2, 6]` the answer is `18`
(take `a[0..4] = 3-1+4-1+5 = 10` and `a[6..7] = 2+6 = 8`).

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, all-negative arrays, arrays mixing
signs with long negative runs, `k = 1`, `k = ceil(n/2)` (the maximum number of segments),
small `n`, and large `n = 2*10^5` with `|a[i]|` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum total value of EXACTLY k disjoint non-empty
    // contiguous segments of a[0..n-1].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

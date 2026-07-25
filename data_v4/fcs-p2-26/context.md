# Maximum-sum strictly increasing subsequence

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be negative). Choose a
**subsequence** — a subset of the positions taken in their original left-to-right order — whose
values are **strictly increasing**, and **maximize the sum** of the chosen values. You may choose
the empty subsequence, so the answer is always at least `0`. Output that maximum sum.

This is the weighted cousin of the longest-increasing-subsequence problem: instead of maximizing the
*count* of elements in a strictly increasing chain, we maximize their *sum*. It is the kind of
subproblem that appears inside scheduling, packing, and sequence-alignment DPs, so getting the
one-dimensional version exactly right — including the negative-value and empty-subsequence corners —
matters.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 5000`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable sum.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [1, 100, 2, 3, 4, 5, 6]` the answer is `101` — the chain `1, 100` already sums to
`101`, and it beats the longer chain `1, 2, 3, 4, 5, 6` (sum `21`). The longest increasing
subsequence is *not* the maximum-sum one.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, the empty
array (`n = 0`), single element (`n = 1`), all-negative arrays, all-equal arrays, strictly decreasing
arrays, and large `n = 5000` with values near `10^9`.

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

    // TODO: compute the maximum sum of a strictly increasing subsequence
    // (values strictly increasing, original order preserved, empty allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

# Weighted inversion sum (sum of products over out-of-order pairs)

## Research question

You are given a sequence of `n` positive integers `a[0..n-1]`. Consider every pair of positions
`(i, j)` with `i < j` that is **out of order**, meaning `a[i] > a[j]` — an *inversion*. For each such
pair, form the product `a[i] * a[j]`, and **sum these products over all inversions**. Output that sum.
If there are no inversions (the sequence is non-decreasing) the answer is `0`.

This generalizes the classic inversion-count problem: instead of counting `1` per out-of-order pair,
each pair is *weighted* by the product of the two values. It is the kind of order-statistic sweep that
shows up inside ranking, similarity, and discordance measures (a product-weighted Kendall-style score).

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^5`); then `n` integers `a[i]`
  (`1 <= a[i] <= 30000`), whitespace-separated.
- Output (stdout): a single line with the weighted inversion sum.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 4, 1, 5]` the inversions are `(3,1)` at positions `(0,1)`, `(3,1)` at `(0,3)`,
and `(4,1)` at `(2,3)`; the answer is `3*1 + 3*1 + 4*1 = 10`.

## Background

The naive definition is a double loop over all pairs — `O(n^2)` — which is `~5*10^9` operations at
`n = 10^5` and will not fit the time limit. The structure to exploit is that we only ever compare each
new element against *earlier* elements, and we only need an aggregate of those earlier elements, not the
list itself. Standard tools for this class of order-statistic sweep include divide-and-conquer accounting
during a merge step, and Fenwick (BIT) / segment-tree sweeps over compressed values.

## Evaluation settings

Judged on hidden tests covering: the empty sequence (`n = 0`), a single element, a strictly increasing
sequence (answer `0`), a strictly decreasing sequence (every pair contributes), sequences with many
ties (equal elements are *not* inversions), and large `n = 10^5` with values near `30000`.

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
    vector<int> a(n);
    for (auto &x : a) cin >> x;

    // TODO: sum a[i]*a[j] over all pairs i<j with a[i] > a[j], in O(n log n).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

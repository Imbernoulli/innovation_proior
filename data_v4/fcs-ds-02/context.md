# k-th smallest value in a subarray (static range k-th order statistic)

## Research question

You are given a fixed array `a[0..n-1]` of integers and a batch of `q` queries. Each query is a triple
`(l, r, k)` (with `1 <= l <= r <= n` and `1 <= k <= r - l + 1`) and asks: **if the elements
`a[l-1], a[l], ..., a[r-1]` were sorted in non-decreasing order, what value sits in position `k`?**
In other words, the `k`-th smallest value of the contiguous block `[l, r]` (1-based, `k` 1-based).
There are **no updates** to the array — it is read once and then only queried.

The challenge is the scale: both `n` and `q` reach `2*10^5`, so an `O((r-l) log(r-l))` sort per query
(re-sorting the slice every time) is far too slow in the worst case. The question is what static
structure answers each range order-statistic query in (poly-)logarithmic time after near-linear
preprocessing.

## Input / output contract

- Input (stdin):
  - Line 1: two integers `n` and `q` (`1 <= n, q <= 2*10^5`).
  - Line 2: `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
  - Next `q` lines: three integers `l r k` per line, `1 <= l <= r <= n`, `1 <= k <= r - l + 1`.
- Output (stdout): `q` lines; line `j` is the `k`-th smallest value of the `j`-th query's subarray.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
7 3
1 5 2 6 3 7 4
2 5 3
1 7 4
3 7 1
```

Expected output:

```
5
4
2
```

(Query `2 5 3`: the slice `[5,2,6,3]` sorts to `[2,3,5,6]`, 3rd smallest is `5`.
Query `1 7 4`: the whole array sorts to `[1,2,3,4,5,6,7]`, 4th is `4`.
Query `3 7 1`: the slice `[2,6,3,7,4]` sorts to `[2,3,4,6,7]`, 1st is `2`.)

## Background

Two families of approach are on the table before committing to one:

- **Re-sort the slice per query.** For each `(l, r, k)`, copy `a[l-1 .. r-1]`, sort it, return the
  `(k-1)`-th element. Trivially correct and `O((r-l) log(r-l))` per query, but with `r-l` and `q` both
  near `2*10^5` the worst case is on the order of `q * n log n ~ 2*10^5 * 2*10^5 * 18`, which is
  astronomically over budget. The open question is how to avoid touching `r-l` elements per query.

- **A static order-statistic structure indexed by value.** If we could, for any prefix length `t`,
  ask "how many of the first `t` elements have value-rank `<= x`?" in `O(log n)`, then the count of
  small values inside a window `[l, r]` would be a difference of two prefix counts, and a binary
  search over the value axis would isolate the `k`-th smallest. The open question is how to store one
  cumulative value-histogram **per prefix** without paying `O(n)` memory for each of the `n` prefixes.

## Evaluation settings

Judged on hidden tests covering: arrays with many duplicate values (including all-equal arrays),
arrays mixing large-magnitude positives and negatives near `±10^9`, single-element arrays (`n = 1`),
queries with `k = 1` and `k = r - l + 1` (the minimum and maximum of the slice), single-position
queries (`l = r`), and large `n = q = 2*10^5` to stress both preprocessing and per-query time.
Values can be negative and need coordinate compression; the answer is always an actual array value.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: build a static structure that answers, for each query (l, r, k),
    // the k-th smallest value of a[l-1 .. r-1] in (poly-)logarithmic time.

    for (int query = 0; query < q; query++) {
        int l, r, k;
        cin >> l >> r >> k;
        long long answer = 0; // TODO
        cout << answer << '\n';
    }
    return 0;
}
```

# Dynamic maximum-sum subarray with point updates

## Research question

You maintain an array `a[0..n-1]` of integers (values may be negative or zero). You must support a
mix of two operations, online, in the order given:

- **Update** `1 i x`: set `a[i] = x`.
- **Query** `2 l r`: report the maximum sum of a **non-empty** contiguous subarray that lies
  entirely inside `a[l..r]` (so `l <= e_start <= e_end <= r`).

Because the chosen subarray must be **non-empty**, an all-negative range must still return its
least-bad single element (a negative number), and a range that is all zeros returns `0`. The empty
subarray is **not** an allowed answer. This is the dynamic version of Kadane's problem: the static
one-shot answer is folklore, but supporting point updates and arbitrary subranges forces a segment
tree whose nodes carry enough information to merge two ranges in `O(1)`. Getting the merge identity
and the sign handling exactly right — especially on the all-negative and all-zero corners — is the
whole difficulty.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `q`
  (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`). The second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`). Then `q` lines follow, each either
  `1 i x` (`0 <= i <= n-1`, `-10^9 <= x <= 10^9`) or `2 l r` (`0 <= l <= r <= n-1`).
- Output (stdout): for each query of type `2`, print the maximum non-empty subarray sum on its own
  line, in the order the queries appear.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `a = [-2, 1, -3, 4, -1, 2]` the query `2 0 5` returns `5` (the subarray `[4, -1, 2]`),
and `2 0 2` returns `1` (the single element `[1]`).

## Background

The static "maximum subarray sum" of a fixed array is solved in `O(n)` by Kadane's scan. But here
the array changes (point updates) and queries hit arbitrary subranges, so a left-to-right scan per
query would be `O(nq)` — up to `4*10^10` operations, far too slow.

Two families of approach are on the table before committing:

- **Re-scan per query.** Answer each `2 l r` with a fresh Kadane scan over `a[l..r]`, and apply each
  update in `O(1)`. Correct and trivial, but `O(nq)` worst case; it only survives if both `n` and
  `q` are tiny. Useful as an independent brute-force oracle, not as the shipped solution.
- **Segment tree of "subarray summaries".** Each node over a range stores four longs: the best
  prefix sum, the best suffix sum, the total sum, and the best subarray sum inside that range. Two
  adjacent summaries combine in `O(1)`, so build is `O(n)`, each update is `O(log n)`, and each
  query is `O(log n)`. The open questions are the exact merge formulas and — the crux — the
  **identity element** used for out-of-range / empty segments so that an empty piece can never be
  mistaken for a valid non-empty subarray.

## Evaluation settings

Judged on hidden tests covering: all-positive ranges; ranges mixing negatives, zeros, and
positives; all-negative ranges (answer is a negative single element); all-zero ranges (answer `0`);
single-element ranges `l == r`; updates that flip a range from positive-containing to all-negative
and back; and large `n, q = 2*10^5` with values near `10^9`, so a range total can reach
`~2*10^14` and must use 64-bit arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Node {
    long long pre, suf, tot, best;
};

int n, q;
vector<Node> tree;
vector<long long> a;

// TODO: define the merge / identity / leaf for the "best subarray" segment tree,
// then build, point-update, and range-query the maximum NON-EMPTY subarray sum.

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    a.resize(n);
    for (auto &x : a) cin >> x;

    cout.flush();
    return 0;
}
```

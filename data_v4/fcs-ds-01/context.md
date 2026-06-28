# Offline range-distinct counting over many queries

## Research question

You are given an array `a[0..n-1]` of integers and a batch of `q` queries. Each
query is a pair `(l, r)` with `0 <= l <= r < n`, and it asks for the number of
**distinct values** that occur in the contiguous block `a[l..r]` (both endpoints
inclusive). All queries are known up front — this is the **offline** setting,
so you are free to answer them in any order you like as long as each result is
reported against its original position.

The naive reading "for each query, walk the range and count distinct" is
`O(q*n)`. At the stated scale that is `2*10^5 * 2*10^5 = 4*10^10` element
touches, which cannot finish in time. There is no per-query sublinear trick that
treats queries in isolation either, because "distinct in a range" is not
decomposable the way a sum is (you cannot subtract two prefix answers — a value
can appear in both prefixes). The interesting question is whether reordering the
queries so that consecutive ones share most of their range lets a single sliding
window answer the whole batch cheaply.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `q`
    (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`).
  - The second line has `n` integers `a[0], ..., a[n-1]`
    (`-10^9 <= a[i] <= 10^9`).
  - Each of the next `q` lines has two integers `l` and `r`
    (`0 <= l <= r < n`), one query per line, **0-indexed and inclusive**.
- Output (stdout): `q` lines. Line `k` is the number of distinct values in
  `a[l_k .. r_k]` for the `k`-th query **in input order**.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
Input
6 3
1 2 1 3 2 1
0 5
1 3
2 2

Output
3
3
1
```

Range `[0,5]` over `[1,2,1,3,2,1]` contains the distinct values `{1,2,3}` → `3`.
Range `[1,3]` is `[2,1,3]` → `{1,2,3}` → `3`. Range `[2,2]` is `[1]` → `1`.

## Background

This is the canonical *offline range-distinct* problem. Two families of approach
are on the table before committing to one:

- **Per-query rescan.** For each `(l, r)` iterate `a[l..r]` and count distinct
  values with a hash set or a freshly cleared boolean array. Obviously correct
  and trivial to write, but it is `Theta(q*n)` in the worst case and the
  per-query set also carries a constant factor. It is fine for `n, q <= 1000`
  and hopeless at `2*10^5`.

- **Reorder the queries and sweep one window.** Keep a single window `[L, R]`
  and a value→count table, with `O(1)` cost to extend or shrink the window by
  one position (a value's count crossing 0 changes the running distinct total).
  If the queries are visited in an order that keeps total window movement small,
  the whole batch is answered in roughly `O((n + q) * sqrt(q))`. The open
  questions are the *exact ordering* that bounds the pointer travel and the
  precise `O(1)` add/remove bookkeeping.

There is also an online `O((n + q) log n)` route (sort queries by `r`, sweep a
Fenwick tree marking each value's last occurrence), but the offline
window-sweep is the standard, simplest tool that comfortably fits these limits,
so that is what the framework below is scaffolded toward.

## Evaluation settings

Judged on hidden tests covering: `n = 1`; all-equal arrays (every answer `1`);
all-distinct arrays (answer `= r - l + 1`); arrays over a tiny alphabet (size 1
and 2, which maximizes the count-table churn where a value's count repeatedly
hits and leaves zero); many identical queries; single-element ranges (`l == r`);
negative values and zeros (so the solution must not assume non-negativity);
values at the `+-10^9` extremes; and full-scale `n = q = 2*10^5` with both
narrow ranges and ranges that span almost the whole array (the adversarial case
for window-movement bounds).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    vector<int> ql(q), qr(q);
    for (int i = 0; i < q; i++) cin >> ql[i] >> qr[i];

    // TODO: answer each query with the count of distinct values in a[l..r],
    // and print the answers in the original input order.
    vector<long long> ans(q, 0);

    string out;
    for (int i = 0; i < q; i++) { out += to_string(ans[i]); out += '\n'; }
    cout << out;
    return 0;
}
```

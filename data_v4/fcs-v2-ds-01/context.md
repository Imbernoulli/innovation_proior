# Range frequency-power queries (sum of squared multiplicities)

## Research question

You are given an array `a[0..n-1]` of positive integers and `q` offline range queries. For a query
`[l, r]` (1-based, inclusive) define, over the multiset `a[l..r]`, the **frequency power**

```
P(l, r) = sum over distinct values v of (count of v in a[l..r])^2.
```

For example, if `a[l..r] = (1, 2, 1, 3, 1, 2)` then value `1` appears 3 times, `2` appears twice, `3`
once, so `P = 3^2 + 2^2 + 1^2 = 14`. Answer every query.

This is the canonical "range frequency statistic" workload: each query depends on the *entire*
frequency table of its window, not on a decomposable aggregate like a sum or a max, so classic
range structures (prefix sums, sparse tables, segment trees over the array) do not apply directly.
The interesting question is how to answer all `q` windows fast when the windows overlap arbitrarily.

## Input / output contract

- Input (stdin):
  - line 1: two integers `n` and `q` (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`).
  - line 2: `n` integers `a[i]` (`1 <= a[i] <= 2*10^5`).
  - next `q` lines: two integers `l r` (`1 <= l <= r <= n`), a 1-based inclusive range.
- Output (stdout): `q` lines; line `k` is `P(l_k, r_k)` for the `k`-th query in input order.
- Time limit: 2 seconds. Memory: 256 MB.

`P` can be as large as `n^2 = 4*10^10`, which exceeds 32-bit range, so the answers must be 64-bit.

## Background

Two angles are on the table before committing to one:

- **Per-query recomputation.** For each `[l, r]`, sweep the window and tally counts in a hash/array,
  then sum the squares. This is `O(n)` per query and obviously correct, but `O(n q)` overall — up to
  `4*10^10` operations at the limits, far too slow. It is the reference oracle, not the solution.
- **Offline window-sliding (Mo's algorithm).** `P` supports an `O(1)` *incremental* update: when one
  element enters or leaves the window, the squared-frequency sum changes by a closed-form delta. So if
  the `q` queries are answered in an order that keeps the total movement of the two window endpoints
  small, each unit of movement costs `O(1)`. The open question is the ordering: a naive order moves the
  endpoints `O(n q)` times, while a careful order brings the total movement down to `O((n + q) sqrt n)`.
  The constant factor of that ordering is exactly what this problem is about.

## Evaluation settings

Judged on hidden tests covering: heavy value collisions (few distinct values, large counts), nearly
all-distinct arrays, single-element queries, full-array queries `[1, n]`, `n = 1`, `q` clustered into
tiny scattered windows, and the maximal `n = q = 2*10^5` with answers near `4*10^10` (so the running
statistic and the outputs must be 64-bit).

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
    vector<int> a(n);
    for (auto &x : a) cin >> x;

    // TODO: read the q ranges, answer P(l, r) = sum of squared frequencies for each,
    // and print the answers in the original query order.

    return 0;
}
```

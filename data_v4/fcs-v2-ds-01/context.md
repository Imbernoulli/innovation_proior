# Range frequency-power queries (sum of squared multiplicities)

## Research question

You are given an array `a[0..n-1]` of positive integers and `q` offline range queries. For a query
`[l, r]` (1-based, inclusive) define, over the multiset `a[l..r]`, the **frequency power**

```
P(l, r) = sum over distinct values v of (count of v in a[l..r])^2.
```

For example, if `a[l..r] = (1, 2, 1, 3, 1, 2)` then value `1` appears 3 times, `2` appears twice, `3`
once, so `P = 3^2 + 2^2 + 1^2 = 14`. Answer every query.

## Input / output contract

- Input (stdin):
  - line 1: two integers `n` and `q` (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`).
  - line 2: `n` integers `a[i]` (`1 <= a[i] <= 2*10^5`).
  - next `q` lines: two integers `l r` (`1 <= l <= r <= n`), a 1-based inclusive range.
- Output (stdout): `q` lines; line `k` is `P(l_k, r_k)` for the `k`-th query in input order.
- Time limit: 2 seconds. Memory: 256 MB.

## Evaluation settings

Judged on hidden tests covering: heavy value collisions (few distinct values, large counts), nearly
all-distinct arrays, single-element queries, full-array queries `[1, n]`, `n = 1`, `q` clustered into
tiny scattered windows, and the maximal `n = q = 2*10^5`.

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

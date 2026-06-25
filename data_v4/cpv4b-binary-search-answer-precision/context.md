# The K-th cheapest reagent pairing

## Research question

A pharmacology lab prepares a reagent by combining exactly one *base* compound with exactly one
*catalyst*. There are `n` candidate bases, where base `i` requires `a[i]` micrograms, and `m`
candidate catalysts, where catalyst `j` requires `b[j]` micrograms. Pairing base `i` with catalyst
`j` consumes `a[i] * b[j]` micrograms of stock — the cost of that pairing.

There are `n * m` possible pairings, each with its own cost. The lab wants the `K`-th cheapest cost
when **all** `n * m` pairing costs are listed in nondecreasing order (costs are compared as plain
numbers; equal costs each occupy their own slot in the ordering). Output that `K`-th smallest cost.

This is the order-statistic-of-a-product-table problem. It is the kind of question that looks like a
sort — and would be, if `n * m` were small — but the table is far too large to materialize, so the
real work is a binary search on the *answer value* with a feasibility test that counts how many
pairings cost at most a candidate threshold. The catch that decides correctness is arithmetic: the
costs and the search bounds run past what a signed 64-bit integer can hold.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `m`, `K`
  (`1 <= n <= 10^5`, `1 <= m <= 10^5`, `1 <= K <= n*m`). The second line has `n` integers `a[i]`
  (`1 <= a[i] <= 4*10^9`). The third line has `m` integers `b[j]` (`1 <= b[j] <= 4*10^9`).
  Tokens are whitespace-separated; line breaks are not significant.
- Output (stdout): a single line with the `K`-th smallest value among the `n*m` products
  `a[i]*b[j]`.
- Time limit: 2 seconds. Memory: 256 MB.

Example. For `n = m = 3`, `K = 8`, `a = [4, 2, 3000000000]`, `b = [3000000000, 5, 2]`, the nine
products sorted are
`[4, 8, 10, 20, 6000000000, 6000000000, 12000000000, 15000000000, 9000000000000000000]`,
so the 8th smallest is `15000000000` (that is `3000000000 * 5`). Note the largest product,
`3000000000 * 3000000000 = 9*10^18`, already sits within a whisker of the signed 64-bit ceiling
(`~9.22*10^18`), and the upper bound of the search range, `4*10^9 * 4*10^9 = 1.6*10^19`, exceeds it.

## Background

All values are positive, so every product is positive and the count of products at most a threshold
`x` is nondecreasing in `x`. That monotonicity is exactly what a binary search on the answer needs.
Two design questions remain before committing:

- **The feasibility test.** For a candidate threshold `x`, how many pairings satisfy
  `a[i]*b[j] <= x`? Summing per base, this is the count of catalysts `j` with `b[j] <= x / a[i]`.
  The open question is whether to compute `x / a[i]` (a division that truncates) or to keep the test
  as a product comparison `a[i]*b[j] <= x`, and what numeric type each choice demands.
- **The search range.** The answer can be as large as `1.6*10^19`, which overflows signed 64-bit and
  also overflows the magnitude where intermediate products are formed. The open question is the
  arithmetic type for the candidate `x`, for the products, and for printing the result.

## Evaluation settings

Judged on hidden tests covering: tiny tables (`n = m = 1`); `K = 1` and `K = n*m` (the extreme
order statistics); tables with many equal products and ties at the boundary value; mixed small and
near-maximal values so the largest products land just below and just above the signed 64-bit ceiling;
and large `n = m = 10^5` random tables with values near `4*10^9` (so the feasibility test and the
printed answer both exceed 64-bit signed range, and a naive division-based test reveals its rounding
behavior).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __int128 i128;

int n, m;
long long K;
vector<u64> a, b;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m >> K)) return 0;
    a.resize(n);
    b.resize(m);
    for (auto &x : a) cin >> x;
    for (auto &x : b) cin >> x;
    sort(b.begin(), b.end());

    // TODO: binary-search the smallest threshold x with (#pairings a[i]*b[j] <= x) >= K,
    //       using an exact-arithmetic feasibility test, and print x.
    i128 answer = 0;

    // ... print answer in base 10 ...
    (void)answer;
    return 0;
}
```

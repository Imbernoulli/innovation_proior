# The K-th cheapest reagent pairing

## Research question

A pharmacology lab prepares a reagent by combining exactly one *base* compound with exactly one
*catalyst*. There are `n` candidate bases, where base `i` requires `a[i]` micrograms, and `m`
candidate catalysts, where catalyst `j` requires `b[j]` micrograms. Pairing base `i` with catalyst
`j` consumes `a[i] * b[j]` micrograms of stock — the cost of that pairing.

There are `n * m` possible pairings, each with its own cost. The lab wants the `K`-th cheapest cost
when **all** `n * m` pairing costs are listed in nondecreasing order (costs are compared as plain
numbers; equal costs each occupy their own slot in the ordering). Output that `K`-th smallest cost.

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
so the 8th smallest is `15000000000` (that is `3000000000 * 5`).

## Evaluation settings

Judged on hidden tests covering: tiny tables (`n = m = 1`); `K = 1` and `K = n*m` (the extreme
order statistics); tables with many equal products and ties at the boundary value; mixed small and
near-maximal values; and large `n = m = 10^5` random tables with values near `4*10^9`.

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

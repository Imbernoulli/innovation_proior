# Number of distinct ways to make a target sum, modulo a prime

## Research question

You are given `n` coin denominations, a target value `S`, and a prime modulus `p`. You have an
**unlimited** supply of each denomination. Count the number of **distinct ways** to choose a multiset
of coins whose values sum to **exactly** `S`, and output that count **modulo `p`**.

"Distinct ways" means order does **not** matter: a way is determined solely by *how many coins of each
denomination value* it uses. Using `1 + 2` and `2 + 1` is the **same** way. A way is fixed by, for each
distinct denomination value, the number of coins of that value used. (So duplicate denomination values
in the input refer to the same coin type and collapse to a single type.)

This is the counting variant of the coin-change problem (sometimes called the "coin change 2" /
unbounded-knapsack counting problem). It is the kind of subroutine that appears in partition counting,
generating-function evaluation, and combinatorial enumeration, where the count can be astronomically
large and is therefore requested modulo a prime.

## Input / output contract

- Input (stdin):
  - The first line contains three integers `n`, `S`, and `p`
    (`1 <= n <= 200`, `0 <= S <= 2*10^5`, `2 <= p <= 10^9 + 7`, `p` prime).
  - The second line contains `n` integers `c[0..n-1]`, the denominations
    (`1 <= c[i] <= 10^6`), whitespace-separated. Denominations may repeat and may exceed `S`.
- Output (stdout): a single line with the number of distinct ways (multisets) to form `S`,
  taken modulo `p`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for denominations `{1, 2, 5}`, `S = 5`, and `p = 10^9 + 7`, the answer is `4`. The four
multisets are `5`, `2 + 2 + 1`, `2 + 1 + 1 + 1`, and `1 + 1 + 1 + 1 + 1`.

## Evaluation settings

Judged on hidden tests covering: small dense coin sets where the count is large, sets with and without
a `1` coin, unreachable targets, `S = 0`, denominations larger than `S`, duplicate denominations in the
input, very small moduli such as `p = 2` and `p = 3`, large primes near `10^9 + 7`, and large instances
with `S = 2*10^5` and `n = 200`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S, p;
    if (!(cin >> n >> S >> p)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // TODO: count the number of distinct multisets of coins (unlimited supply of
    // each denomination, order does not matter) summing to exactly S, modulo p.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

# Counting stamp combinations for exact postage

## Research question

A philately shop sells stamps in `n` denominations. Denomination `i` has face value `v[i]` (a positive
integer) and the shop currently holds `c[i]` copies of it in stock. A customer needs to assemble stamps
whose face values add up to **exactly** a required postage `S`. Two ways of paying are considered the
**same** if they use the same number of stamps of every denomination — only the multiset of stamps
matters, never the order in which they are placed on the envelope, and stamps of equal value within one
denomination are interchangeable.

Count how many **distinct** ways the customer can make exactly `S` using at most `c[i]` copies of each
denomination `i`. The count can be astronomically large, so report it **modulo a given integer `MOD`**.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `S`, `MOD`
  (`1 <= n <= 200`, `0 <= S <= 2*10^5`, `1 <= MOD <= 10^9`). Then `n` lines follow; line `i` has two
  integers `v[i]` and `c[i]` (`1 <= v[i] <= 10^9`, `0 <= c[i] <= 10^9`).
- Output (stdout): a single line with the number of distinct exact-postage combinations, taken modulo
  `MOD`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `S = 10`, `MOD = 1000000007` and denominations `(v, c) = (2, 3), (3, 2), (5, 1)`,
the answer is `2`. The two combinations are `{2, 2, 3, 3}` (two 2-stamps and two 3-stamps) and
`{2, 3, 5}` (one of each). No other multiset within the stock limits sums to `10`.

## Evaluation settings

Judged on hidden tests covering: tiny hand-checkable instances; unreachable targets; `S = 0`;
denominations with value greater than `S`; very large supply limits `c[i]`; `MOD = 1`; non-prime
moduli; and full-scale inputs with `n = 200`, `S = 2*10^5`, and large `c[i]`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    long long MOD;
    if (!(cin >> n >> S >> MOD)) return 0;

    vector<long long> v(n), c(n);
    for (int i = 0; i < n; i++) cin >> v[i] >> c[i];

    // TODO: count distinct multisets of stamps (at most c[i] of denomination i)
    // whose face values sum to exactly S, modulo MOD.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

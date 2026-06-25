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

This is a bounded-knapsack *counting* problem. The headline trap is double-counting: the same multiset
of stamps can be reached by many different orders of "adding a stamp," and an innocent-looking DP loop
order silently counts those orders as distinct payments.

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

## Background

The constraint that each denomination has a limited supply `c[i]` makes this *bounded* (not unbounded)
knapsack, and the requirement to count *multisets* rather than *sequences* is what makes the loop
structure load-bearing. Two framings are on the table before committing:

- **Capacity-outer DP.** Define `f[s]` = number of ways to reach sum `s`, fill it by, for each `s`,
  adding any one denomination whose value fits. This is the shortest code, but it is the classic
  shape that counts *ordered* compositions: reaching `5` as `2+3` and as `3+2` would both be tallied.
  The open question is whether (and how) this over-counts.
- **Denomination-outer DP.** Process denominations one at a time, and for each, decide how many copies
  `k` (from `0` to `c[i]`) to use. Each multiset is then produced exactly once, in canonical "by
  denomination" order. The open questions are the exact bounded transition and how to evaluate it fast
  enough when `c[i]` is large.

## Evaluation settings

Judged on hidden tests covering: tiny hand-checkable instances; unreachable targets (answer `0`);
`S = 0` (the empty combination, answer `1`); denominations with value greater than `S`; very large
supply limits `c[i]` (effectively unbounded, stressing the fast bounded transition); `MOD = 1` (answer
always `0`); non-prime moduli (so no modular inverse may be assumed); and full-scale inputs with
`n = 200`, `S = 2*10^5` and large `c[i]` (so an `O(S * sum of c[i])` transition is too slow and 32-bit
intermediate products would overflow).

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

# Divisor summatory function via the Dirichlet hyperbola method

## Research question

Let `d(k)` denote the number of positive divisors of `k`. For a given integer `n`, compute the
**divisor summatory function**

```
D(n) = d(1) + d(2) + ... + d(n) = sum_{k=1}^{n} d(k).
```

Equivalently, `D(n) = sum_{i=1}^{n} floor(n / i)`, because a fixed `i` divides exactly
`floor(n/i)` of the integers in `1..n`, so counting divisor-incidences by the divisor `i`
rather than by the dividend `k` gives the same total. The task is to output `D(n)` for a single
`n` that can be as large as `10^12`, well past the range where an explicit `O(n)` loop over
`i = 1..n` finishes in time.

This is the canonical setting for the **Dirichlet hyperbola method**: counting lattice points
`(a, b)` with `a * b <= n` by exploiting the symmetry of the region `a * b <= n` about the line
`a = b`.

## Input / output contract

- Input (stdin): a single integer `n` with `0 <= n <= 10^12`.
- Output (stdout): a single line with `D(n)`, the value of `sum_{k=1}^{n} d(k)`.
  By convention `D(n) = 0` for `n <= 0` (an empty sum).
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6` the divisor counts are `d(1..6) = 1, 2, 2, 3, 2, 4`, so the answer is
`1 + 2 + 2 + 3 + 2 + 4 = 14`.

## Background

Two representations of the same quantity are on the table before committing to an algorithm:

- **Sum over dividends.** `D(n) = sum_{k=1}^{n} d(k)`. Computing each `d(k)` by trial division
  costs `O(sqrt(k))`, and a divisor sieve costs `O(n log n)` time and `O(n)` memory. Both are
  obviously correct but scale with `n`, which is fatal at `n = 10^12`.
- **Sum over divisors.** `D(n) = sum_{i=1}^{n} floor(n / i)`. This is a single `O(n)` loop with
  `O(1)` memory — much better, but still linear in `n`, so still far too slow at `10^12` (a
  trillion iterations).

The open question is whether the `sum floor(n/i)` form hides enough structure to be evaluated in
sublinear time. The relevant structural fact is geometric: `D(n)` counts the lattice points
`(a, b)` with `a >= 1`, `b >= 1`, and `a * b <= n` (each `k <= n` contributes one point per
divisor pair). That region sits under a hyperbola and is symmetric across `a = b`.

## Evaluation settings

Judged on hidden tests covering: the convention region `n = 0` (answer `0`); the smallest
nontrivial values `n = 1, 2, 3`; perfect squares `n = s*s` (where the symmetry correction term is
sharpest); values straddling a perfect square (`s*s - 1`, `s*s`, `s*s + 1`); a spread of
mid-sized `n`; and the maximum `n = 10^12`, where any `O(n)` approach times out and where the
returned value (`~2.8 * 10^13`) must not overflow 32-bit arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n;
    if (!(cin >> n)) return 0;
    if (n <= 0) { cout << 0 << "\n"; return 0; }

    // TODO: compute D(n) = sum_{k=1..n} d(k) = sum_{i=1..n} floor(n/i)
    // fast enough for n up to 1e12 (an O(n) loop over i is too slow).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```

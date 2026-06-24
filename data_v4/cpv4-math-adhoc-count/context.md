# Counting coprime-spread pairs by least common multiple

## Research question

You are given a single integer `n`. Count the number of **unordered pairs** `{x, y}` of integers with
`1 <= x < y <= n` whose least common multiple does not exceed `n`:

```
count = #{ {x, y} : 1 <= x < y <= n,  lcm(x, y) <= n }.
```

Each unordered pair is counted **once** (the pair `{2, 3}` and `{3, 2}` are the same object and must not
both be tallied). Output that count.

This is the kind of multiplicative-counting subproblem that hides a double-count: the clean way to reason
about `lcm(x, y)` is to factor out `g = gcd(x, y)` and write `x = g*a`, `y = g*b` with `a, b` coprime, but
the moment you parameterize by `(g, a, b)` you have to decide exactly once whether `(a, b)` and `(b, a)`
describe the same pair — and getting that decision off by a factor of two, or off by one on the diagonal,
is the whole game.

## Input / output contract

- Input (stdin): a single integer `n` with `0 <= n <= 10^6`.
- Output (stdout): a single line with the number of qualifying unordered pairs.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6` the answer is `9`. The qualifying pairs are
`{1,2}, {1,3}, {1,4}, {1,5}, {1,6}, {2,3}, {2,4}, {2,6}, {3,6}` — each has `lcm <= 6`.

## Background

The brute force is a double loop over `x < y` computing `lcm(x, y) = x / gcd(x, y) * y` and testing
`<= n`. That is `O(n^2 log n)` and only survives for `n` in the hundreds; for `n = 10^6` it is hopeless.

Two ideas are on the table before committing to one:

- **Direct double loop, optimized.** Loop `x` from `1`, `y` from `x+1`, break early when `lcm` exceeds
  `n`. Still fundamentally quadratic because for `x = 1` every `y <= n` qualifies, so it alone is `O(n)`
  and the whole thing degrades badly. The open question is whether any pruning rescues it (it does not).
- **Reparameterize by gcd.** Every pair with `gcd = g` is `x = g*a`, `y = g*b` with `gcd(a, b) = 1` and
  `lcm = g*a*b`. Counting then becomes a sum over coprime `(a, b)` of how many `g` keep `g*a*b <= n`,
  i.e. `floor(n / (a*b))`. The open question is the exact index ranges — which `(a, b)` to enumerate so
  that every unordered `{x, y}` is hit exactly once and `x = y` is never produced.

## Evaluation settings

Judged on hidden tests covering: tiny `n` (`0`, `1`, `2`, `3`) where the answer is `0` or `1`; small `n`
checked against the brute force; mid-range `n` in the thousands; and large `n = 10^6` where the answer is
in the tens of millions (so the accumulator must be 64-bit) and an `O(n^2)` method times out.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

static long long gcdll(long long a, long long b) {
    while (b) { long long t = a % b; a = b; b = t; }
    return a;
}

int main() {
    long long n;
    if (!(cin >> n)) return 0;

    // TODO: count unordered pairs {x, y}, 1 <= x < y <= n, with lcm(x, y) <= n,
    // each pair counted exactly once.
    long long ans = 0;

    cout << ans << "\n";
    return 0;
}
```

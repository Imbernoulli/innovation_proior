# Counting coprime-spread pairs by least common multiple

## Research question

You are given a single integer `n`. Count the number of **unordered pairs** `{x, y}` of integers with
`1 <= x < y <= n` whose least common multiple does not exceed `n`:

```
count = #{ {x, y} : 1 <= x < y <= n,  lcm(x, y) <= n }.
```

Each unordered pair is counted **once** (the pair `{2, 3}` and `{3, 2}` are the same object and must not
both be tallied). Output that count.

## Input / output contract

- Input (stdin): a single integer `n` with `0 <= n <= 10^6`.
- Output (stdout): a single line with the number of qualifying unordered pairs.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6` the answer is `9`. The qualifying pairs are
`{1,2}, {1,3}, {1,4}, {1,5}, {1,6}, {2,3}, {2,4}, {2,6}, {3,6}` — each has `lcm <= 6`.

## Background

The brute force is a double loop over `x < y` computing `lcm(x, y) = x / gcd(x, y) * y` and testing
`<= n`. That is `O(n^2 log n)` and only survives for `n` in the hundreds; for `n = 10^6` it is hopeless.

## Evaluation settings

Judged on hidden tests covering: tiny `n` (`0`, `1`, `2`, `3`) where the answer is `0` or `1`; small `n`
checked against the brute force; mid-range `n` in the thousands; and large `n = 10^6` where an `O(n^2)`
method times out.

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

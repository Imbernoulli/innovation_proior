# Counting capped multisets of a fixed size, modulo a prime

## Research question

A confectioner stocks `n` distinct flavours of candy. A "gift box" is an unordered selection of
exactly `k` candies (a multiset over the flavours), but the machine that fills a box can dispense
**at most `c` candies of any single flavour** — so every flavour appears between `0` and `c` times
in a box, and the counts across all flavours sum to `k`.

Count the number of distinct gift boxes, modulo the prime `p = 1 000 000 007`.

Equivalently: count the integer tuples `(x_1, ..., x_n)` with `0 <= x_i <= c` for every `i` and
`x_1 + ... + x_n = k`. With no upper cap this is the classic stars-and-bars value
`C(n + k - 1, n - 1)`; the cap `c` turns it into a bounded-composition count, the small inclusion-
exclusion identity that is *very* easy to get subtly wrong in either the summation range or the
binomial indices.

## Input / output contract

- Input (stdin): three integers `n k c` on one line (whitespace-separated):
  - `0 <= n <= 2*10^6` (number of flavours),
  - `0 <= k <= 2*10^6` (box size), with `n + k <= 4*10^6`,
  - `0 <= c <= 2*10^6` (per-flavour cap).
- Output (stdout): a single line with the number of distinct boxes, taken modulo `1 000 000 007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `k = 4`, `c = 2` the answer is `6`. The valid count-tuples summing to `4` with
each entry in `0..2` are `(2,2,0),(2,0,2),(0,2,2),(2,1,1),(1,2,1),(1,1,2)`.

## Background

Two pieces of standard machinery combine here.

- **Stars and bars.** The number of nonnegative integer solutions to `x_1 + ... + x_m = r` is
  `C(m + r - 1, m - 1)` for `m >= 1`. The `m = 0` case is a genuine corner: there is exactly one
  solution (the empty tuple) when `r = 0`, and none otherwise — the binomial formula does not cover
  it.
- **Inclusion-exclusion to enforce the upper cap.** Start from the uncapped count and subtract the
  arrangements where some flavour exceeds the cap. Forcing a chosen flavour to use at least `c + 1`
  candies removes `c + 1` from the budget; choosing *which* `j` flavours are forced gives a `C(n, j)`
  factor and an alternating sign. The exact summation limit (when does a term first vanish) and the
  exact binomial indices are where the counting goes wrong.

The two open questions before committing: what is the precise alternating sum, and where does it
stop so that nothing is double-counted or counted with the wrong sign.

## Evaluation settings

Judged on hidden tests covering: tiny boxes (`k = 0`, the empty box, always counts as `1`); no
flavours (`n = 0`); a zero cap (`c = 0`, so only the empty box is reachable); cases where `k` exceeds
`n*c` (answer `0`, nothing fits); the unconstrained regime `c >= k` (answer equals the plain
stars-and-bars value); boundary boxes where `k` is an exact multiple of `c + 1`; and large inputs
with `n + k` up to `4*10^6` (so factorial precomputation and modular inverses must be `O(n + k)` and
fit in memory).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

long long power_mod(long long b, long long e, long long m) {
    b %= m; if (b < 0) b += m;
    long long r = 1 % m;
    while (e > 0) {
        if (e & 1) r = (__int128)r * b % m;
        b = (__int128)b * b % m;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k, c;
    if (!(cin >> n >> k >> c)) return 0;

    // TODO: count size-k multisets over n flavours with each flavour used at most c times, mod p.
    long long answer = 0;

    cout << answer % MOD << "\n";
    return 0;
}
```

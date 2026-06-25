# Smallest chisel that splits every boulder within the blow budget

## Research question

A quarry crew must break `n` boulders down for transport. Boulder `i` weighs `w[i]` (a positive
integer, in kilograms). The crew owns one pneumatic chisel whose **power** `p` is an integer: a single
blow can separate off a chunk of at most `p` kilograms, so to reduce a boulder of weight `w` to pieces
that each weigh **at most** `p` you must keep chiselling until no piece exceeds `p`. The most economical
way to do that for one boulder needs `ceil(w / p)` final pieces, hence `ceil(w / p) - 1` blows.

The crew is allowed a total of `k` blows across **all** boulders combined (the chisel can only fire so
many times before the compressor must cool down). A bigger `p` means fewer blows per boulder, but the
crew wants the **least powerful chisel** that still fits inside the blow budget, because higher-power
chisels are far more expensive to rent.

Find the minimum integer power `p >= 1` such that the total number of blows needed to make every
boulder consist of pieces of weight at most `p` is at most `k`.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `k`
  (`1 <= n <= 2*10^5`, `0 <= k <= 2*10^14`). The second line holds `n` integers `w[i]`
  (`1 <= w[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the minimum chisel power `p`.
- Time limit: 1 second. Memory: 256 MB.

Example: for boulders `w = [10, 7, 8, 5]` and budget `k = 6`, the answer is `4`. With `p = 4` the blow
counts are `ceil(10/4)-1 = 2`, `ceil(7/4)-1 = 1`, `ceil(8/4)-1 = 1`, `ceil(5/4)-1 = 1`, totalling
`5 <= 6`. With `p = 3` they are `3 + 2 + 2 + 1 = 8 > 6`, so `3` is infeasible and `4` is the minimum.

## Background

The predicate "power `p` keeps the total blows within budget" is **monotone in `p`**: increasing `p`
can only reduce the blows for each boulder, so once a power is feasible every larger power is feasible
too. That monotonicity is exactly what lets us binary-search the smallest feasible `p` instead of
scanning every candidate from `1` upward.

Two design choices have to be nailed down before any code is written, and both are boundary questions:

- **The per-boulder blow formula.** Reducing weight `w` to pieces of size at most `p` needs
  `ceil(w / p)` pieces and therefore `ceil(w / p) - 1` blows. The off-by-one lives in whether a boulder
  whose weight is an exact multiple of `p` needs one fewer or one more blow than the naive guess — the
  "`<= p`" (inclusive) cap is what fixes it.
- **The binary-search boundary.** We want the *minimum* feasible `p`, so the search must collapse the
  `[lo, hi]` window toward the smallest value that passes the predicate; moving the wrong endpoint, or
  using `mid - 1` where `mid` is itself a candidate, lands one step off the true answer.

## Evaluation settings

Judged on hidden tests covering: a single boulder; `k = 0` (no blows allowed, forcing `p` up to the
heaviest boulder); budgets that land exactly on a divisibility boundary (`w` an exact multiple of the
answer `p`); many equal weights; and large instances with `n = 2*10^5` and `w[i]` near `10^9` (so the
blow total reaches `~2*10^14` and must be held in a 64-bit accumulator).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> w(n);
    long long maxw = 1;
    for (auto &x : w) { cin >> x; maxw = max(maxw, x); }

    // TODO: binary-search the minimum chisel power p in [1, maxw] such that the
    // total number of blows, sum over boulders of (ceil(w[i]/p) - 1), is <= k.
    long long answer = maxw;

    cout << answer << "\n";
    return 0;
}
```

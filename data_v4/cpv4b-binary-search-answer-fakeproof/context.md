# The lattice quota gate

## Research question

A warehouse robot lives on the positive integer lattice. A *parcel slot* is an ordered pair of
positive integers `(a, b)` with `a >= 1` and `b >= 1`; you may think of `a` as a shelf index and `b`
as a bin index within that shelf. A slot `(a, b)` is *reachable within budget `m`* when its
"footprint" `a * b` does not exceed `m`, i.e. `a * b <= m`.

Let `R(m)` be the number of reachable slots for budget `m`:

```
R(m) = #{ (a, b) : a >= 1, b >= 1, a * b <= m }.
```

For a target quota `K`, you must report the **smallest** budget `m >= 1` such that `R(m) >= K`
(at least `K` slots become reachable). You answer several independent quotas `K` per run.

`R` is non-decreasing in `m`, so the budget that first reaches the quota is well defined; this is a
"smallest `m` with a monotone predicate" question, i.e. binary search on the answer. The whole
difficulty is evaluating the predicate `R(m) >= K` fast and *correctly*, because `m` can be on the
order of `10^11` and you cannot enumerate slots.

## Input / output contract

- Input (stdin): the first token is `q` (`1 <= q <= 50`), the number of quotas. Then `q` integers
  `K` follow, one per line, each `1 <= K <= 10^12`, whitespace-separated.
- Output (stdout): for each quota, on its own line, the smallest `m >= 1` with `R(m) >= K`.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
input
5
1
2
5
100
1000000000000

output
1
2
3
28
40677885960
```

(`R(1) = 1` so `K = 1` gives `m = 1`; `R(2) = 3 >= 2` while `R(1) = 1 < 2`, so `K = 2` gives `m = 2`;
`R(3) = 5 >= 5` while `R(2) = 3 < 5`, so `K = 5` gives `m = 3`; for `K = 100` the answer is `28`; for
the trillion-quota the answer is `40677885960`.)

## Background

There is a clean reformulation. For a fixed shelf index `a`, the bin indices `b` with `a * b <= m`
are exactly `b = 1, 2, ..., floor(m / a)`, so shelf `a` contributes `floor(m / a)` reachable slots,
and

```
R(m) = sum_{a = 1}^{m} floor(m / a).
```

That sum has `m` terms, which is far too many to add directly when `m ~ 10^11`. The same quantity
also equals `sum_{t = 1}^{m} d(t)`, where `d(t)` is the number of divisors of `t` (each slot
`(a, b)` with `a * b = t <= m` is one divisor pair of some `t`), the *divisor summatory function*.

The standard way to evaluate `R(m)` in sublinear time is the **hyperbola method**: the lattice
points under the hyperbola `a * b <= m` are symmetric across the diagonal `a = b`, so you can count
the points with the smaller coordinate up to `floor(sqrt(m))`, double, and subtract the square block
you counted twice. Translating that picture into an exact closed form in terms of `s = floor(sqrt(m))`
is the crux — and it is the kind of bound that is very tempting to assert from memory and very easy
to get wrong by one term.

## Evaluation settings

Judged on hidden tests covering: the smallest quotas (`K = 1, 2, 3`), quotas that land exactly on a
jump of `R` versus between jumps, moderate quotas where a one-term error in the closed form first
shows up, and the largest quotas (`K = 10^12`, answer near `4 * 10^10`) with up to `q = 50` of them
in a single run so an `O(sqrt(m))` predicate per binary-search step is required.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

// R(m) = #{ (a,b) : a>=1, b>=1, a*b <= m }.
long long R(long long m) {
    // TODO: evaluate R(m) in O(sqrt(m)) using the hyperbola identity,
    //       with the square-block correction term derived and CHECKED, not assumed.
    return 0;
}

int main() {
    int q;
    if (scanf("%d", &q) != 1) return 0;
    while (q--) {
        long long K;
        scanf("%lld", &K);
        // TODO: binary-search the smallest m >= 1 with R(m) >= K.
        printf("%lld\n", 0LL);
    }
    return 0;
}
```

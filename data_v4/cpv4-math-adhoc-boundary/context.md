# Counting divisors-of-a-pile: how many shovel widths leave a quotient in a band

## Research question

A pile of `n` identical bricks is to be packed into boxes of some integer width `x`. With width `x`
each box holds exactly `x` bricks, so the number of *completely filled* boxes is `floor(n / x)` (the
last partial box, if any, is ignored). The foreman only cares about widths that leave the number of
filled boxes inside an acceptance band `[a, b]`.

You are asked: for given `n`, `a`, `b`, how many integer widths `x` with `1 <= x <= n` satisfy

```
a <= floor(n / x) <= b ?
```

You must answer `q` independent such questions. This is a `math-adhoc` counting problem: the honest
`O(n)` scan per query is too slow at the stated scale, so the intended solution turns the two
quotient inequalities into index bounds on `x` and counts the integers in the resulting interval in
`O(1)`. The whole difficulty is getting those bounds *inclusive/exclusive-correct* — one stray
off-by-one and a single value of `x` at each endpoint is wrongly counted or dropped.

## Input / output contract

- Input (stdin): the first token is `q` (`1 <= q <= 2*10^5`), the number of queries. Then `q` lines
  follow, each with three integers `n a b`:
  - `1 <= n <= 10^12`,
  - `0 <= a <= b <= 10^12` in the well-formed tests, though your code should also behave for the
    degenerate orderings described below.
- Output (stdout): `q` lines, the `i`-th being the count of widths `x in [1, n]` with
  `a <= floor(n/x) <= b`.
- Time limit: 1 second. Memory: 256 MB.

Example. For the input

```
8
10 1 10
10 2 3
10 1 1
7 0 0
1 1 1
1000000000000 1000000000000 1000000000000
6 7 9
100 3 3
```

the correct output is

```
10
3
5
0
1
1
0
8
```

Walk the first few: with `n = 10`, every `x in [1,10]` gives `floor(10/x)` between `1` and `10`, so
all `10` widths qualify. For the band `[2,3]` only `x in {3,4,5}` work (`floor(10/3)=3`,
`floor(10/4)=floor(10/5)=2`), giving `3`. For the band `[1,1]`, `x in {6,7,8,9,10}` all give
`floor(10/x)=1`, giving `5`. For `n=7, [0,0]` no width in `[1,7]` ever yields quotient `0`, so `0`.

## Background

The function `x -> floor(n/x)` is non-increasing on `x in [1, n]`: as the box gets wider, the number
of full boxes can only drop or stay equal. That monotonicity is what lets each one-sided quotient
constraint become a one-sided constraint on `x`:

- `floor(n/x) >= a` is a "quotient at least `a`" condition. It bounds `x` from **above**.
- `floor(n/x) <= b` is a "quotient at most `b`" condition. It bounds `x` from **below**.

So the set of valid `x` is an interval, and the count is its length. The catch — and the entire
point of this problem — is that the two endpoints have *different* inclusivity. The `>= a` side turns
into a closed upper bound `x <= floor(n/a)`; the `<= b` side turns into an *open* lower bound
`x > floor(n/(b+1))` (note the `b+1`, not `b`). Mixing these up, or using `floor(n/b)` instead of
`floor(n/(b+1))`, miscounts by exactly one width near the boundary, which the hidden tests target
directly with single-value bands like `[k, k]`.

The identity behind the upper bound: for integers `n, x, a >= 1`, `floor(n/x) >= a` iff `x <= n/a`
iff `x <= floor(n/a)`. The identity behind the lower bound: `floor(n/x) <= b` iff `n/x < b+1` iff
`x > n/(b+1)` iff `x > floor(n/(b+1))` (the last step because `x` is an integer). These two standard
floor identities are the load-bearing facts.

## Evaluation settings

Judged on hidden tests covering: full-range bands `[1, n]` and `[0, n]`; single-value bands `[k, k]`
(the off-by-one magnets, including `k = 1` where the interval reaches all the way to `x = n`, and
large `k` near `n`); empty answers (bands above the maximum possible quotient `n`, or the impossible
`[0,0]`); `n = 1`; `n` up to `10^12` with bands touching `10^12` (so quotients and the product
`b+1` must be handled as 64-bit); and up to `2*10^5` queries (so each query must be `O(1)`, not a
scan over `x`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count x in [1, n] with a <= floor(n/x) <= b.
long long solve(long long n, long long a, long long b) {
    // TODO: turn the two quotient inequalities into inclusive/exclusive index
    //       bounds on x and return the size of the resulting interval.
    return 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, a, b;
        cin >> n >> a >> b;
        cout << solve(n, a, b) << "\n";
    }
    return 0;
}
```

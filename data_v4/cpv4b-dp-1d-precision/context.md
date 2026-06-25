# Cheapest pace over a mandatory relay

## Research question

A relay course is laid out as `n` stones in a straight line, numbered `0` to `n-1`. A runner starts
on stone `0` and must finish on stone `n-1`. From the stone she is on, stone `i`, she may make exactly
one of two kinds of hop:

- a **short hop** to stone `i+1`, which costs effort `e1[i]` and covers distance `g1[i]`
  (available for every `i` from `0` to `n-2`);
- a **long hop** to stone `i+2`, which costs effort `e2[i]` and covers distance `g2[i]`
  (available for every `i` from `0` to `n-3`).

All efforts and distances are **positive** integers. A *plan* is any sequence of hops that takes her
from stone `0` to stone `n-1` (she may never overshoot past `n-1`). The **pace** of a plan is

```
pace = (total effort of the hops used) / (total distance of the hops used).
```

She wants to move as economically as possible: **minimize the pace** over all valid plans. Because the
course must be connected from `0` to `n-1`, she cannot simply grab one cheap hop — the plan has to span
the whole line, so the answer is a genuine blend of many hops, not a single edge.

Output the minimum achievable pace as an **exact reduced fraction** `P/Q` with `Q > 0`.

## Input / output contract

- Input (stdin):
  - the first token is `n` (`2 <= n <= 2*10^5`);
  - then `n-1` lines, the `i`-th of which (for `i = 0 .. n-2`) holds two integers `e1[i] g1[i]`
    — the effort and distance of the short hop from stone `i` to stone `i+1`;
  - then `n-2` lines, the `i`-th of which (for `i = 0 .. n-3`) holds two integers `e2[i] g2[i]`
    — the effort and distance of the long hop from stone `i` to stone `i+2`.
  - All efforts and distances satisfy `1 <= e, g <= 10^9`.
- Output (stdout): one line with two integers `P Q`, the minimum pace `P/Q` in lowest terms (`Q > 0`).
- Time limit: 2 seconds. Memory: 256 MB.

Example. For

```
4
3 1
5 2
4 1
2 3
6 5
```

the four plans are `0->1->2->3` (effort `12`, distance `4`, pace `3`), `0->1->3` (effort `9`, distance
`6`, pace `3/2`), and `0->2->3` (effort `6`, distance `4`, pace `3/2`). The minimum pace is `3/2`, so
the answer is `3 2`.

## Background

Minimizing a ratio `E(plan)/D(plan)` of two sums is **fractional (linear-fractional) optimization**. The
key obstacle is that a ratio is not additive: you cannot run an ordinary shortest-path DP directly on
`e/g` per hop, because the best ratio over a path is *not* the combination of the best ratios of its
parts. Two routes are on the table before committing to one:

- **Greedy / per-hop ratio.** At each stone, take whichever hop (short or long) has the smaller `e/g`.
  This is `O(n)` and trivial, but a locally cheap ratio can force an expensive continuation, so the open
  question is whether it is ever optimal.
- **Parametric (Dinkelbach) search with an inner DP.** For a trial value `λ`, ask whether some plan has
  `E - λ·D < 0`; this is a plain additive shortest path with hop weight `e - λ·g`, which *is* a clean
  `dp-1d` because hops only go forward by `1` or `2`. Sweeping `λ` to the point where the cheapest plan
  has `E - λ·D = 0` pins the optimum exactly. The open questions are the recurrence, how to keep `λ`
  rational so the answer is exact, and — because the distances accumulate over the whole course — how
  large the integers in the weight `e·Q - g·P` can get.

## Evaluation settings

Judged on hidden tests covering: tiny courses (`n = 2`, a single forced hop; `n = 3`), courses where the
all-short-hop plan is far from optimal, courses with many near-equal hop ratios that force exact
comparison, and large courses `n = 2*10^5` with efforts and distances near `10^9` (so the accumulated
totals reach `~2*10^14` and the products inside the comparison overflow 64-bit arithmetic).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<ll> e1(max(0, n-1)), g1(max(0, n-1));
    vector<ll> e2(max(0, n-2)), g2(max(0, n-2));
    for (int i = 0; i < n-1; i++) scanf("%lld %lld", &e1[i], &g1[i]); // short hop i -> i+1
    for (int i = 0; i < n-2; i++) scanf("%lld %lld", &e2[i], &g2[i]); // long  hop i -> i+2

    // TODO: minimize (sum of effort)/(sum of distance) over all plans 0 -> n-1,
    //       and print it as a reduced fraction "P Q" (Q > 0).
    ll P = 0, Q = 1;

    printf("%lld %lld\n", P, Q);
    return 0;
}
```

# The k-th lit minute of a lantern festival

## Research question

A night market strings up `m` types of programmable lanterns over the main avenue. Lantern type `i`
is configured with a period `p_i` (in minutes) and switches on at every minute that is a multiple of
its period: minutes `p_i, 2*p_i, 3*p_i, ...`. A given minute `t >= 1` is called **lit** if *at least
one* lantern type switches on at minute `t` — that is, if `t` is divisible by at least one of the
periods.

The festival organizers want to know when the **`k`-th lit minute** occurs, counting lit minutes in
increasing order of time, so they can schedule the headline act. Given the `m` periods and the index
`k`, output the time (in minutes) of the `k`-th lit minute.

Two periods may be equal, and one period may be a multiple of another — the same physical minute must
never be counted twice no matter how many lantern types happen to light it. The whole difficulty of
the problem is in counting *distinct* lit minutes correctly.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `k`
  (`1 <= m <= 10`, `1 <= k <= 10^12`). The second line has `m` integers `p_1 ... p_m`
  (`1 <= p_i <= 10^6`), whitespace-separated. Periods are not necessarily distinct.
- Output (stdout): a single line with the time of the `k`-th lit minute.
- Time limit: 1 second. Memory: 256 MB.

The answer fits in a signed 64-bit integer: the `k`-th lit minute is at most `k * min(p_i) <=
10^12 * 10^6 = 10^18`, which is within the `~9.2*10^18` range of `long long`.

Example: for `m = 2`, `k = 10`, periods `p = [2, 3]`, the lit minutes in order are
`2, 3, 4, 6, 8, 9, 10, 12, 14, 15, ...`, so the 10th lit minute is at time `15`.

## Background

The answer is monotone in a way that invites **binary search on the answer**: define
`f(x) =` the number of lit minutes in the range `[1, x]`. As `x` grows, `f(x)` is non-decreasing, and
the `k`-th lit minute is the smallest `x` with `f(x) >= k`. So if we can evaluate `f(x)` quickly, we
binary-search `x` over `[1, 2*10^18]`.

Evaluating `f(x)` is a counting problem: *how many integers in `[1, x]` are divisible by at least one
of `p_1, ..., p_m`?* Two ideas are on the table before committing:

- **Sum of individual counts.** Add up `floor(x / p_i)` over all `i`. This is `O(m)` per query and
  trivial to write. The open question is whether it counts each lit minute exactly once.
- **Inclusion–exclusion over subsets.** For every non-empty subset `S` of the periods, a minute is
  divisible by *all* of `S` exactly when it is a multiple of `lcm(S)`; alternating-sign over subset
  sizes counts the union without double counting. This is `O(2^m)` per query (here `2^10 = 1024`),
  using least common multiples. The open questions are the sign convention and overflow of the LCM.

## Evaluation settings

Judged on hidden tests covering: a single lantern (`m = 1`); periods that are pairwise coprime (no
overlap); periods with heavy overlap and divisibility chains (e.g. `2, 4, 8`); duplicate periods
(e.g. `6, 6`); the boundary where the answer approaches `10^18`; and `m = 10` with large `k` so the
`O(2^m * log(answer))` work and the LCM overflow guard both get exercised.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m;
ll k;
vector<ll> p;

// count of integers in [1, x] divisible by at least one p_i (distinct minutes).
ll countLit(ll x) {
    // TODO: count distinct lit minutes in [1, x] without double counting.
    return 0;
}

int main() {
    if (!(cin >> m >> k)) return 0;
    p.resize(m);
    for (auto &v : p) cin >> v;

    ll lo = 1, hi = (ll)2e18;
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLit(mid) >= k) hi = mid; else lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
```

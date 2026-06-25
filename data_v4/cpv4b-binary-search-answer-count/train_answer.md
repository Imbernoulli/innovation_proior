**Problem.** `m` lantern types have periods `p_1, ..., p_m`; type `i` lights at every multiple of
`p_i`. A minute `t >= 1` is *lit* if it is divisible by at least one period. Counting *distinct* lit
minutes in increasing order, output the time of the `k`-th lit minute. Constraints: `m <= 10`,
`p_i <= 10^6`, `k <= 10^12`; periods may repeat.

**Key idea — binary search on the answer with an inclusion–exclusion count.** Let
`f(x)` = number of distinct lit minutes in `[1, x]` = number of integers in `[1,x]` divisible by at
least one period. `f` is non-decreasing, so the `k`-th lit minute is the smallest `x` with
`f(x) >= k`; binary-search `x` over `[1, 2*10^18]`. Compute `f(x)` by inclusion–exclusion over the
`2^m - 1` non-empty subsets of periods: a minute divisible by *all* periods in subset `S` is a
multiple of `lcm(S)`, so

```
f(x) = sum over non-empty S of  (-1)^(|S|+1) * floor(x / lcm(S))
```

odd-sized subsets add, even-sized subtract. With `m <= 10` this is at most `1023` terms per query.

**Why the obvious shortcut is wrong.** The one-liner `f(x) = sum_i floor(x / p_i)` double-counts any
minute divisible by several periods. On `p = [2, 3]`, `x = 6`, the true lit minutes are `2,3,4,6`
(four of them) but the sum gives `floor(6/2)+floor(6/3) = 5` — minute `6` is counted twice. That
double-count propagates: for `k = 10` the buggy count crosses `10` at `x = 12` and returns `12`,
while the correct answer is `15`. Inclusion–exclusion fixes it: `{2}:+3, {3}:+2, {2,3}:-1` gives
`4`, removing the one double-counted minute.

**Pitfalls.**
1. *Double counting.* The whole problem is counting *distinct* minutes; naive per-period sums
   overcount shared multiples. Use inclusion–exclusion with `lcm`, not a plain sum.
2. *LCM overflow.* Multiplying LCMs blindly overflows `long long` once four-plus coprime periods near
   `10^6` are combined (`lcm ≈ 10^24`). Guard every multiply: before `lcmv *= step` check
   `lcmv > x / step`; if so the subset's LCM exceeds `x`, its term `floor(x/lcm)` is `0`, so skip it.
   This both prevents overflow and short-circuits useless subsets. The test must be strict `>` so the
   boundary subset with `lcm == x` (which contributes a real `+1`) is kept — a `>=` here silently
   restores the double-count.
3. *Search bound.* The answer can reach `k * min(p_i) = 10^12 * 10^6 = 10^18`, so `hi` must exceed
   `10^18` (use `2*10^18`, still `< LLONG_MAX`); a reflexive `hi = 10^9` makes the boundary case
   unreachable. Use `mid = lo + (hi-lo)/2` to avoid `lo+hi` overflow.

**Edge cases.** `m = 1` reduces to multiples of `p[0]` (`f(x) = floor(x/p[0])`). Duplicate periods
(e.g. `[6,6]`) deduplicate automatically: `lcm(6,6)=6` makes the `+2 -1` collapse to one
`floor(x/6)`, so no manual dedup is needed. Divisibility chains (`[2,4,8]`) collapse to
`f(x)=floor(x/2)` as the redundant terms cancel. Boundary answer `10^18` is reachable and prints
exactly in 64 bits.

**Complexity.** `O(2^m * m)` per count and `O(log(hi))` iterations: about `2^10 * 10 * 61 ≈
6*10^5` operations — far under the 1-second limit. `O(m)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;

int m;
ll k;
vector<ll> p;

// count of integers in [1, x] divisible by at least one p_i,
// via inclusion-exclusion over subsets, using lcm with an overflow cap.
ll countLit(ll x) {
    ll total = 0;
    for (int mask = 1; mask < (1 << m); mask++) {
        // build lcm of the chosen p_i; if it exceeds x, its contribution is 0
        // so we can stop early and treat the term as 0.
        ll lcmv = 1;
        bool overflow = false;
        for (int i = 0; i < m; i++) {
            if (mask & (1 << i)) {
                ll g = __gcd(lcmv, p[i]);
                ll step = p[i] / g;            // lcmv * step is the new lcm
                // overflow / over-x guard: if lcmv * step > x, this subset
                // contributes floor(x / lcm) == 0, so mark and bail.
                if (lcmv > x / step) { overflow = true; break; }
                lcmv *= step;
            }
        }
        if (overflow) continue;                // term is 0
        int bits = __builtin_popcount(mask);
        ll term = x / lcmv;
        if (bits & 1) total += term;           // odd-sized subset: add
        else total -= term;                    // even-sized subset: subtract
    }
    return total;
}

int main() {
    if (!(cin >> m >> k)) return 0;
    p.resize(m);
    for (auto &v : p) cin >> v;

    // Binary search for the smallest x with countLit(x) >= k.
    ll lo = 1, hi = (ll)2e18;
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLit(mid) >= k) hi = mid;
        else lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
```

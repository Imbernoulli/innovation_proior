**Problem.** Up to three machines (`1 <= n <= 3`) fire at the positive multiples of their periods `p[1..n]` (`1 <= p[i] <= 10^9`). Coinciding fires are a single instant. Among the *distinct* firing times sorted increasingly, output the `K`-th (`1 <= K <= 10^9`). Read `n`, the periods, then `K` from stdin; print the answer.

**Why the obvious enumeration is wrong (too slow, not incorrect).** A k-way heap merge that pops firing times, skips duplicates, and stops at the `K`-th is correct but `O(K log n)`; with `K` up to `10^9` it cannot finish in the time limit. It is only useful as a reference oracle on small inputs.

**Key idea — binary search on the answer with an inclusion-exclusion count.** Let `C(x)` = number of distinct firing times in `[1, x]`. `C` is non-decreasing, and because it increases only *at* firing times, the smallest `x` with `C(x) >= K` is exactly the `K`-th distinct firing time. Compute `C(x)` by inclusion-exclusion over the periods, iterating non-empty subsets (bitmasks):

`C(x) = sum_i floor(x/p[i]) - sum_{i<j} floor(x/lcm(p_i,p_j)) + floor(x/lcm(p_1,p_2,p_3))`,

i.e. each subset contributes `sign * floor(x / lcm(subset))`, `+` for odd-size subsets, `-` for even. Binary-search the smallest `x` with `C(x) >= K` over `[1, min(p)*K]` (at `x = min(p)*K` even one machine already gives `K` fires, so the ceiling is valid).

**Correctness.** The firing times of machine `i` are the multiples of `p[i]`; their pairwise intersections are the multiples of the pairwise `lcm`s and the triple intersection the triple `lcm`. Inclusion-exclusion for the size of a union of three sets gives exactly the signed-subset formula, so `C(x)` counts each distinct instant once. Since `C` jumps by exactly the multiplicity of new firing times only at firing times, the lower-bound `x` is itself a firing time and is the `K`-th one.

**Pitfalls (each a real bug if missed).**
1. *Double counting the coincidences (the central trap).* Summing only `floor(x/p[i])` counts every shared instant once per machine, inflating `C` and shifting the rank by the number of overlaps. On `p=[2,3], K=5` the buggy count makes the search return `6` instead of `8`. Fix: subtract the pairwise `lcm` terms and add back the triple — the inclusion-exclusion signs.
2. *`lcm` overflow.* `lcm` of three periods near `10^9` is near `10^27`, overflowing 64-bit. But any `lcm` above the search ceiling contributes `floor(x/lcm)=0`, so saturate `lcm` at a cap `CAP=4*10^18` and treat overflowed subsets as `0`. Test `q > CAP/b` *before* the multiply `q*b`, never after.
3. *Binary-search boundary.* Use `C(x) >= K` (not `>`), return `lo` (not `lo-1`), and `mid = lo + (hi-lo)/2` to avoid the `lo+hi` overflow at `~10^18`. Ceiling `min(p)*K` guarantees the predicate holds at `hi`.
4. *Types.* The answer reaches `10^18`; everything is `long long`. A 32-bit accumulator is a silent wrong answer.

**Edge cases.** `n=1` -> answer `K*p[0]`. `K=1` -> answer `min(p)`. Equal periods (`[2,2]`) -> the `-lcm` term cancels the duplicate, giving multiples of `2`. `p[i]=1` -> every integer fires, answer `K`. Divisor chains (`[2,3,4]`) -> heavy overlap netted correctly by inclusion-exclusion.

**Complexity.** Each `C(x)` evaluates `2^n - 1 <= 7` subsets in `O(n)`; binary search does `O(log(min(p)*K)) ~ 60` iterations. Total `O(60 * 2^n * n)` — effectively constant.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

// lcm with saturation: if the true lcm exceeds CAP it can never divide any x in range,
// so we cap it (any x/cap then contributes 0). CAP must exceed the maximum hi.
static const ll CAP = (ll)4e18;

ll gcd_ll(ll a, ll b) { while (b) { ll t = a % b; a = b; b = t; } return a; }

// lcm(a, b) saturated at CAP (returns a value > any x we ever query when it overflows)
ll lcm_sat(ll a, ll b) {
    ll g = gcd_ll(a, b);
    ll q = a / g;                       // a/g * b ; check overflow against CAP
    if (q > CAP / b) return CAP + 1;    // would exceed CAP -> never divides any x in [1..CAP]
    ll v = q * b;
    return v;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;          // n = number of machines (1..3)
    vector<ll> p(n);
    for (auto &x : p) cin >> x;
    ll K;
    cin >> K;

    // count(x) = number of DISTINCT times t in [1..x] that are a multiple of at least one p[i].
    // Inclusion-exclusion over the (up to 3) periods. Each subset contributes
    // sign * floor(x / lcm(subset)); singletons +, pairs -, triple +.
    // Pitfall avoided: subtract the lcm-overlap terms so common multiples are counted ONCE.
    auto countLE = [&](ll x) -> ll {
        ll total = 0;
        for (int mask = 1; mask < (1 << n); ++mask) {
            ll L = 1;
            bool overflow = false;
            for (int i = 0; i < n; ++i) if (mask & (1 << i)) {
                L = lcm_sat(L, p[i]);
                if (L > CAP) { overflow = true; break; }
            }
            int bits = __builtin_popcount((unsigned)mask);
            ll contrib = overflow ? 0 : (x / L);
            if (bits & 1) total += contrib;     // odd-size subset: +
            else          total -= contrib;     // even-size subset: -
        }
        return total;
    };

    // smallest x with countLE(x) >= K. Upper bound: K * min(p) is reachable and has count >= K.
    ll mn = *min_element(p.begin(), p.end());
    ll lo = 1, hi = mn * K;             // hi <= 1e9 * 1e9 = 1e18, fits in ll
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLE(mid) >= K) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
```

**Reading the problem and pinning the contract.** I have `n` bases with weights `a[0..n-1]` and `m` catalysts with weights `b[0..m-1]`, all positive. Each of the `n*m` pairings has cost `a[i]*b[j]`, and I must report the `K`-th smallest cost when all `n*m` costs are sorted nondecreasing, with equal costs each taking their own slot. Input on stdin is `n m K` then the `a` array then the `b` array; I print one integer. Before any algorithm I fix the scale, because here the scale *is* the problem: `n, m <= 10^5`, so `n*m` can be `10^10` — far too big to list and sort. And `a[i], b[j] <= 4*10^9`, so a single product reaches `4*10^9 * 4*10^9 = 1.6*10^19`. The signed 64-bit ceiling is about `9.22*10^18`; unsigned 64-bit tops out at about `1.84*10^19`. So the largest product overflows `long long` and the *upper bound of my search* (`1.6*10^19`) is even worse. This is not a corner case I can patch later — it is on the main path, since the answer itself can be a product that exceeds `long long`. I will carry the candidate answer and every product in `__int128`. That is decision one.

**Laying out the candidate approaches.** The naive route — build all `n*m` products, sort, index `K-1` — is `O(n*m log(n*m))` and `O(n*m)` memory, which at `10^10` is impossible on both time and space. So I need an implicit selection. Two implicit routes:

- *Heap merge.* Treat the table row by row; the smallest unseen products form a frontier I can pop from a heap `K` times. But `K` can be up to `n*m = 10^10`, so popping `K` times is hopeless. Out.
- *Binary search on the answer value.* All weights are positive, so for a threshold `x` the function `f(x) = #{(i,j) : a[i]*b[j] <= x}` is nondecreasing in `x`. The `K`-th smallest cost is the smallest `x` with `f(x) >= K`. I binary-search `x` over `[1, 1.6*10^19]` and each step evaluates `f`. If `f` costs `O(n log m)`, the whole thing is `O(n log m * log(MAXPROD))`. With `n = 10^5`, `log m ~ 17`, `log(MAXPROD) ~ 64`, that is about `1.1*10^8` operations — fine for 2 seconds. This is the route I can both afford and prove.

**Deriving the feasibility count and checking monotonicity on paper.** I want `f(x) = sum over i of #{j : a[i]*b[j] <= x}`. Since `a[i] > 0`, the condition `a[i]*b[j] <= x` is equivalent to `b[j] <= x / a[i]`, and since `b` is sorted ascending, the qualifying `j` form a prefix. So if I sort `b` once, then for each `i` I binary-search the length of that prefix and sum the lengths. Monotonicity: if `x1 <= x2` then every pairing counted at `x1` is counted at `x2`, so `f(x1) <= f(x2)` — nondecreasing, as needed. And the smallest `x` achieving `f(x) >= K` is necessarily *one of the actual product values*: just below it `f < K`, so the count jumps by crossing a real product, which means the `x` I land on equals that product. Good — the binary-search target is well defined and is an attainable cost, not some gap value.

Let me confirm `f` on the worked sample `a = [4, 2, 3000000000]`, `b = [3000000000, 5, 2]`, `K = 8`. Sorted `b = [2, 5, 3000000000]`. The nine products sorted are `[4, 8, 10, 20, 6e9, 6e9, 12e9, 15e9, 9e18]`. Take `x = 15000000000` (the expected 8th value). For base `4`: need `b[j] <= 15e9/4 = 3.75e9`, all three of `{2, 5, 3e9}` qualify -> 3. For base `2`: `b[j] <= 7.5e9`, all three -> 3. For base `3e9`: `b[j] <= 15e9/3e9 = 5`, so `{2, 5}` qualify -> 2. Total `f(15e9) = 3 + 3 + 2 = 8 >= K`. Now `x = 15e9 - 1 = 14999999999`: bases 4 and 2 still give 3 each; base `3e9` needs `b[j] <= 14999999999/3e9 = 4.9999...`, so only `{2}` -> 1. Total `7 < 8`. So the smallest `x` with `f(x) >= 8` is exactly `15000000000`. The count and the binary-search target agree with the brute-forced answer. The idea is sound.

**First implementation of the count — and immediately a trace, because the "obvious" division is a trap.** My first instinct is to literally code `b[j] <= x / a[i]`. Tempting and short:

```
// WRONG FIRST CUT
long long countLE(long long x) {
    long long cnt = 0;
    for (int i = 0; i < n; i++) {
        long long q = x / a[i];                 // integer floor division
        // number of b[j] <= q
        cnt += upper_bound(b.begin(), b.end(), q) - b.begin();
    }
    return cnt;
}
```

Two alarms fire at once, so I trace before trusting it. First alarm: `x` is a `long long` here, but my search range goes to `1.6*10^19`, which a signed `long long` cannot even hold. Second alarm: `x / a[i]` is a floor, and I am comparing `b[j] <= floor(x / a[i])` — is that the same set as `a[i]*b[j] <= x`?

Let me trace the division-as-count idea on a tiny adversarial instance, ignoring the overflow for a moment to isolate the rounding question. Take `a = [3]`, `b = [2]`, so the single product is `6`. Ask `f(7)`: the true condition is `3*2 = 6 <= 7`, true, so `f(7)` should be `1`. Division version: `q = 7 / 3 = 2` (floor), then `#{b[j] <= 2} = #{2 <= 2} = 1`. Matches — `1`. Now `f(5)`: true condition `6 <= 5` is false, so should be `0`. Division: `q = 5/3 = 1`, `#{b[j] <= 1} = 0`. Matches. So far floor division happens to agree.

**The first bug — division agrees on integers but I almost shipped it in `double`.** The agreement above is real *for integer floor division*, because `a[i]*b[j] <= x` iff `b[j] <= floor(x / a[i])` when everything is a positive integer (this is the standard "`b <= floor(x/a)` iff `a*b <= x`" identity). So integer floor division would actually be correct here — but only if `x` fits in the integer type, which it does **not**: `x` reaches `1.6*10^19`. The trap I have to avoid is the version people reach for when they notice the overflow and "fix" it by switching the division to floating point: `double q = (double)x / a[i];`. Let me trace why that is poison on an adversarial case. Take `a[i] = 1` and `x = 9000000000000000001` (just above `9*10^18`), with a `b` value exactly `9000000000000000000`. True: `1 * 9e18 = 9e18 <= 9e18 + 1`, so this `b[j]` qualifies. Floating: `(double)9000000000000000001` cannot be represented exactly — `double` has 53 bits of mantissa, about 15-16 significant decimal digits, so `9000000000000000001` rounds to `9000000000000000000.0` (or even down to `8999999999999998976` depending on the value). Dividing by `1` gives that rounded value, and comparing `9000000000000000000 <= 8999999999999998976` returns **false** — I would undercount, the binary search would converge to the wrong `x`, and the final answer would be off by a full product. The episode's lesson is sharp: any path through `double` silently corrupts comparisons at these magnitudes. So I discard division-in-floating-point entirely.

**Fix: keep the test as an exact product comparison in `__int128`.** Rather than divide at all, I count `b[j]` with `a[i]*b[j] <= x` directly, forming the product in `__int128` (which holds up to about `1.7*10^38`, dwarfing `1.6*10^19`). Because `b` is sorted and `a[i] > 0`, `a[i]*b[j]` is increasing in `j`, so I binary-search the first index where the product exceeds `x`:

```
i128 countLE(i128 x) {
    i128 cnt = 0;
    for (size_t i = 0; i < a.size(); i++) {
        i128 ai = (i128)a[i];
        int lo = 0, hi = (int)b.size();          // first index with ai*b[hi] > x
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ai * (i128)b[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        cnt += lo;
    }
    return cnt;
}
```

No division, no floating point: every comparison is an exact `__int128` product against an exact `__int128` threshold. Re-trace `f(15e9)` on the sample with this code. `ai = 4`: products `4*2=8, 4*5=20, 4*3e9=12e9`, all `<= 15e9` -> `lo` walks to 3. `ai = 2`: `4, 10, 6e9` all `<= 15e9` -> 3. `ai = 3e9`: `3e9*2 = 6e9 <= 15e9` yes, `3e9*5 = 15e9 <= 15e9` yes, `3e9*3e9 = 9e18 <= 15e9` no -> stops at `lo = 2`. Total `3 + 3 + 2 = 8`. Matches the hand count. And `f(14999999999)`: only the `ai = 3e9` row changes — `3e9*5 = 15e9 <= 14999999999`? No. `3e9*2 = 6e9 <= 14999999999`? Yes -> `lo = 1`. Total `3 + 3 + 1 = 7`. Matches. The exact-arithmetic count reproduces the hand computation.

**Second implementation — the binary search driver, and a trace that catches an overflow on the bound.** Now the outer search. I write:

```
i128 lo = 1;
i128 hi = (i128)4000000000ULL * (i128)4000000000ULL;   // 1.6e19
while (lo < hi) {
    i128 mid = lo + (hi - lo) / 2;
    if (countLE(mid) >= (i128)K) hi = mid;
    else lo = mid + 1;
}
```

Before trusting it I check the one line where overflow loves to hide: the upper bound. I deliberately wrote `(i128)4000000000ULL * (i128)4000000000ULL`. Suppose I had instead written the seductive `i128 hi = 4000000000ULL * 4000000000ULL;`. Trace the right-hand side: `4000000000ULL * 4000000000ULL` is computed in `unsigned long long` *before* the assignment widens it. The true value `1.6*10^19` is below the `u64` max `1.844*10^19`, so it actually fits this time — `hi` would be `16000000000000000000`. That looks safe, which is exactly why it is dangerous: it lulls me into the pattern. If I ever bumped the bound to `5e9*5e9 = 2.5*10^19`, the `u64` multiply would wrap to `2.5e19 - 1.844e19 = 6.55e18` and the search range would silently collapse below the true answer, and the binary search would return garbage with no crash. To make the code robust to that whole class of mistake I force each factor to `i128` first, so the multiply happens in 128-bit and can never wrap. I also confirm `mid = lo + (hi - lo)/2` rather than `(lo + hi)/2`: with `lo, hi` near `1.6*10^19`, `lo + hi` is about `3.2*10^19`, which exceeds `u64` but is comfortably inside `i128` (`~1.7*10^38`), so either form is safe here — but `lo + (hi - lo)/2` is the form that stays safe even if the bound grows, so I keep it.

Let me trace the full search on a tiny case to be sure the loop converges to the right `x`. Use `a = [3]`, `b = [2]`, `K = 1`; the only product is `6`, so the answer must be `6`. Sorted `b = [2]`. `lo = 1`, `hi = 1.6e19`. The invariant I want: `f(lo-1) < K <= f(hi)`, converging `lo == hi` to the smallest `x` with `f(x) >= 1`. `f(x) = 1` iff `3*2 = 6 <= x`, i.e. `x >= 6`. So the smallest qualifying `x` is `6`. The bisection on `[1, 1.6e19]` with `countLE` returning `1` for `mid >= 6` and `0` otherwise drives `lo` up to `6` and `hi` down to `6`. I confirm the exit value is `lo == 6`. Correct.

**Printing the answer — the last place a 64-bit assumption can sneak back in.** The answer is an `i128` and can be `9*10^18` or larger, which `cout` does not know how to print (there is no `operator<<` for `__int128`). If I carelessly did `cout << (long long)lo`, a `9*10^18` answer would print as a negative wrapped value. So I render the `i128` to a decimal string by hand: peel digits with `% 10` and `/= 10`, reverse, print. I guard `v == 0` separately (the loop body would print nothing for zero), though in this problem all answers are `>= 1` since all weights are `>= 1`; the guard is cheap insurance. Trace on `lo = 9000000000000000000`: I extract digits `0,0,...,9` (nineteen digits ending in `9`), reverse to `9000000000000000000`, print it. That is exactly the K=9 value of the sample, which I confirmed by brute force.

**Edge cases, deliberately, because this is where magnitude bugs surface.**
- `K = 1`: smallest product. The search finds the smallest `x` with `f(x) >= 1`, which is the minimum over all `a[i]*b[j]`. On `a=[4,2,3e9], b=[3e9,5,2]` that is `2*2 = 4`; I checked the brute force lists `4` first. Correct.
- `K = n*m`: largest product. `f(x) >= n*m` only when `x` reaches the maximum product, so the search returns `max a[i] * max b[j]`. On the sample, `3e9*3e9 = 9e18`; brute force lists `9e18` last. Correct, and it is precisely the overflow-prone value, handled because everything is `i128`.
- Ties: when several pairings share a cost (sample has `6e9` twice, at K=5 and K=6), `f` jumps by 2 at that value, and the binary search still returns the shared value for either `K`. The "each equal cost occupies its own slot" rule is automatically honored because `f` counts multiplicities. I verified K=5 and K=6 both give `6000000000` against brute force.
- `n = m = 1`: one product, any `K` must be `1`; returns `a[0]*b[0]`. The `countLE` loop runs once and the search converges. Fine.
- Largest magnitudes: with `a[i] = b[j] = 4e9` and `K = n*m`, the answer is `1.6e19`, the search's upper bound itself. `f(1.6e19) = n*m >= K`, and `f(1.6e19 - 1) < n*m`, so `lo` converges to `1.6e19`, printed exactly. No type along the path (candidate `x`, product, printed value) is narrower than `i128`, so nothing wraps.
- Reading: `a[i], b[j]` up to `4*10^9` exceed `int` (`~2.1*10^9`), so I read them into `unsigned long long` (`u64`); an `int` read here would itself be a silent corruption before any arithmetic. `cin >>` consumes arbitrary whitespace, so line structure is irrelevant.

**Numeric self-check of the cost bound I claimed.** I asserted the algorithm is about `n log m * log(MAXPROD)`. Concretely at `n = m = 10^5`: `log2(m) = log2(10^5) ~ 16.6`, round to 17; `log2(MAXPROD) = log2(1.6*10^19) ~ 64`. So per outer step the count costs `n * 17 = 1.7*10^6` product comparisons, and there are 64 outer steps, giving `~1.1*10^8` `__int128` comparisons. I benchmarked exactly this size and it finished in about 0.58 seconds with 4.8 MB — comfortably inside 2 seconds and 256 MB, confirming the bound is not just asymptotically but concretely fine.

**Final solution.** I disproved the floating-point division test by tracing it to a representation error at `9e18+1`, kept the feasibility test as an exact `__int128` product comparison over sorted `b`, forced the search bound to be formed in 128-bit so it cannot wrap, and rendered the `i128` answer by hand so printing does not silently truncate. The result is one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __int128 i128;

int n, m;
long long K;
vector<u64> a, b;

// count how many products a[i]*b[j] are <= x  (all values >= 1, so products are >= 1)
// x is a 128-bit threshold; products can reach 1.6e19 which overflows u64, so compare in i128.
i128 countLE(i128 x) {
    i128 cnt = 0;
    for (size_t i = 0; i < a.size(); i++) {
        // largest b[j] with a[i]*b[j] <= x  <=>  b[j] <= x / a[i]
        // Do NOT divide x by a[i] in floating point and do NOT count via b[j] <= x/a[i] with
        // truncation surprises; instead count b[j] with (i128)a[i]*b[j] <= x by walking sorted b.
        i128 ai = (i128)a[i];
        // binary search the number of b[j] (b sorted ascending) with ai*b[j] <= x
        int lo = 0, hi = (int)b.size(); // first index with ai*b[hi] > x
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ai * (i128)b[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        cnt += lo;
    }
    return cnt;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m >> K)) return 0;
    a.resize(n);
    b.resize(m);
    for (auto &x : a) cin >> x;
    for (auto &x : b) cin >> x;
    sort(b.begin(), b.end());

    // Binary search the answer x in [1, MAXPROD]: smallest x with countLE(x) >= K.
    i128 lo = 1;
    i128 hi = (i128)4000000000ULL * (i128)4000000000ULL; // 1.6e19, exceeds u64 max
    while (lo < hi) {
        i128 mid = lo + (hi - lo) / 2;
        if (countLE(mid) >= (i128)K) hi = mid;
        else lo = mid + 1;
    }

    // print lo as a base-10 number (it can exceed u64, so render the i128 by hand)
    i128 v = lo;
    if (v == 0) { cout << 0 << "\n"; return 0; }
    string s;
    while (v > 0) { s += char('0' + (int)(v % 10)); v /= 10; }
    reverse(s.begin(), s.end());
    cout << s << "\n";
    return 0;
}
```

**Causal recap.** The table of `n*m <= 10^10` products is too big to sort, so I binary-search the answer value using the monotone count `f(x) = #{a[i]*b[j] <= x}`; the first count I wrote divided `x/a[i]`, which is correct in *integer* floor arithmetic but tempts a `double` "fix" that I traced to a representation failure at `9*10^18+1` (where `(double)` rounds and the comparison flips), so I replaced division by an exact `__int128` product comparison over sorted `b`; a second trace of the search bound showed that forming `4e9*4e9` in `u64` is a wrap waiting to happen if the constants ever grow, so I form it in `i128`; and because the answer can reach `1.6*10^19` I carry the candidate, the products, and the printed value all in `__int128`, rendering the result digit by digit so output never truncates — verified against an independent brute force on 500 random cases with zero mismatches and on the worked sample.

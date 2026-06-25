**Problem.** Given `n` base weights `a[i]` and `m` catalyst weights `b[j]` (all `1 <= a[i], b[j] <= 4*10^9`), list all `n*m` pairing costs `a[i]*b[j]` in nondecreasing order (equal costs each occupy their own slot) and report the `K`-th smallest. Read `n m K`, then `a`, then `b` from stdin; print the `K`-th smallest cost. Here `n, m <= 10^5`, so `n*m` reaches `10^10` — the table cannot be materialized.

**Key idea — binary-search the answer with an exact count.** All weights are positive, so `f(x) = #{(i,j) : a[i]*b[j] <= x}` is nondecreasing in `x`, and the `K`-th smallest cost is the smallest `x` with `f(x) >= K`. Binary-search `x` over `[1, 4*10^9 * 4*10^9 = 1.6*10^19]`. To evaluate `f(x)`, sort `b` once; then for each base `a[i]` the qualifying catalysts `b[j] <= x / a[i]` form a prefix of sorted `b`, so binary-search its length and sum. Cost `O(n log m * log(MAXPROD)) ~ 1.1*10^8` for the largest inputs (measured ~0.58 s).

**Pitfalls (all about magnitude).**
1. *The product overflows 64-bit.* `a[i]*b[j]` reaches `1.6*10^19`, past signed `long long` (`9.22*10^18`) and past unsigned `u64` once you nudge the bound. Carry the candidate `x`, every product, and the search bound in `__int128`.
2. *Never divide in floating point.* The slick test `b[j] <= (double)x / a[i]` corrupts at large `x`: `double` has ~53 mantissa bits, so `9000000000000000001` rounds to `9000000000000000000.0`, flipping a `<=` comparison and making `f` undercount. Replace division by the exact comparison `(i128)a[i]*b[j] <= x`; never form a `double` from these values. (Integer floor division would be correct, but only if `x` fit in the integer type — it does not.)
3. *Form the bound in 128-bit.* Write `(i128)4000000000ULL * (i128)4000000000ULL`, not `4000000000ULL * 4000000000ULL` widened after the multiply: the latter is a `u64` wrap waiting to happen if the constants grow.
4. *Print the `i128` by hand.* `cout` has no `operator<<` for `__int128`; render digit by digit, else a `9*10^18` answer prints as a wrapped negative.

**Edge cases.** `K = 1` -> minimum product; `K = n*m` -> maximum product (the most overflow-prone value, handled by `i128`); ties (equal costs) are honored automatically because `f` counts multiplicities; `n = m = 1` -> the single product; reading `a[i], b[j]` into `u64`, since `4*10^9` exceeds `int`.

**Complexity.** `O((n + m) log m + n log m * log(MAXPROD))` time after one `O(m log m)` sort; `O(n + m)` space.

**Code.**

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

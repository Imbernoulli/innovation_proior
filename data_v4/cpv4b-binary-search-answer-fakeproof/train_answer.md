**Problem.** For a budget `m`, let `R(m) = #{(a, b) : a, b >= 1, a*b <= m}` be the number of positive
lattice points under the hyperbola `a*b <= m`. Given several quotas `K` (`1 <= K <= 10^12`,
`q <= 50`), report for each the smallest budget `m >= 1` with `R(m) >= K`. Read `q` and the `K`s from
stdin, print one `m` per line.

**Key idea — binary search on `m` with a sublinear, *verified* predicate.** `R` is non-decreasing in
`m`, so `R(m) >= K` flips false→true once; binary-search the leftmost true `m`. The predicate needs
`R(m)` fast. Note `R(m) = sum_{a=1}^{m} floor(m/a)` (shelf `a` admits bins `1..floor(m/a)`), but that
is `O(m)`. Use the **hyperbola identity**: with `s = floor(sqrt(m))`, count points with `a <= s`,
double for the symmetric `b <= s` half, and subtract the `s x s` block counted twice (every pair with
`a, b <= s` satisfies `a*b <= s*s <= m`, and no point has both `a, b > s` since `(s+1)^2 > m`):

```
R(m) = 2 * sum_{i=1}^{s} floor(m/i) - s*s,   s = floor(sqrt(m)).
```

This is `O(sqrt(m))` per call. Binary search with `lo = 1`, `hi = 7*10^10` (where `R(hi) ~ 1.76*10^12
>= K` for all legal `K`).

**Pitfalls.**
1. *Asserting the correction term.* The subtracted block is `s*s`, **not** `s`. The `-s` variant is a
   plausible misremembering and it *agrees on `m = 1, 2, 3`* (there `s = s*s = 1`), so a lazy check
   "confirms" it — then it fails at the first `m` with `s >= 2`. Derive the term by inclusion-exclusion
   and *check it numerically against the definition*: `R(4) = 4+2+1+1 = 8`; `2*(4+2) - s*s = 12-4 = 8`
   (correct), while `2*(4+2) - s = 12-2 = 10` (wrong). Run this comparison over thousands of `m`
   before trusting the formula.
2. *Binary-search loop shape.* Pair `hi = mid` with the **strict** `lo < hi`. Using `lo <= hi` with
   `hi = mid` stalls: when `lo == hi == mid` and the predicate holds, `hi = mid` never shrinks the
   interval — infinite loop. Trace `K = 2`: with `lo < hi`, end-game `lo=1,hi=2` → `mid=1`,
   `R(1)=1<2` → `lo=2=hi`, exit, answer `2`.
3. *Inexact `sqrt`.* The whole identity hinges on `s` being *exactly* `floor(sqrt(m))`; `sqrtl` may
   round. Correct it with integer comparisons (decrement while `r*r > m`, increment while
   `(r+1)^2 <= m`).
4. *Overflow.* At `m ~ 7*10^10`, `2*acc ~ 1.76*10^12` and `s*s ~ 6.9*10^10` — fine in `long long`
   (ceiling `~9.2*10^18`), but use `long long` throughout; `int` would silently wrap.

**Edge cases.** `K = 1` → `m = 1` (since `R(1) = 1`, `R(0) = 0`); start `lo = 1`, never `0`. Quotas
that land between jumps of `R` (e.g. `K = 4`, with `R(2)=3 < 4 < 5 = R(3)`) and exactly on a jump
(`K = 5 → m = 3`) are both handled by "leftmost true" with the `>= K` predicate. Perfect-square `m`
(`m = 4, 9, ...`) is exactly where `s*s` matters; the exact integer `isqrt` keeps the block count
right. Boundary at the top: `R(40677885960) = 1000000000079 >= 10^12 > 999999999951 = R(40677885959)`.

**Complexity.** `O(sqrt(m_max))` per predicate, `O(log m_max)` predicates per quota:
`O(q * log(m_max) * sqrt(m_max))`, about `5*10^8` cheap operations total (~0.7 s), `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// floor(sqrt(x)) for x >= 0, exact (guards against long double rounding).
long long isqrt_ll(long long x) {
    if (x <= 0) return 0;
    long long r = (long long)sqrtl((long double)x);
    while (r > 0 && r * r > x) r--;
    while ((r + 1) * (r + 1) <= x) r++;
    return r;
}

// D(m) = #{ (a,b) : a >= 1, b >= 1, a*b <= m }
//      = sum_{t=1..m} d(t)   where d(t) = number of divisors of t.
// Hyperbola / Dirichlet identity with s = floor(sqrt(m)):
//   D(m) = 2 * sum_{i=1..s} floor(m/i) - s*s.
long long D(long long m) {
    if (m <= 0) return 0;
    long long s = isqrt_ll(m);
    long long acc = 0;
    for (long long i = 1; i <= s; i++) acc += m / i;
    return 2 * acc - s * s;
}

int main() {
    int q;
    if (scanf("%d", &q) != 1) return 0;
    while (q--) {
        long long K;
        if (scanf("%lld", &K) != 1) break;
        // Find the smallest m >= 1 with D(m) >= K.
        // D is non-decreasing; D(1) = 1 so K >= 1 guarantees an answer.
        // For K <= 1e12 the answer m < 7e10, where D(7e10) > 1.7e12 >= K.
        long long lo = 1, hi = 70000000000LL; // 7e10
        while (lo < hi) {
            long long mid = lo + (hi - lo) / 2;
            if (D(mid) >= K) hi = mid;
            else lo = mid + 1;
        }
        printf("%lld\n", lo);
    }
    return 0;
}
```

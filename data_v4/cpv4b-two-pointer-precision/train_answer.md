**Problem.** There are `n` amplifier modules; module `i` has a positive integer gain `g[i]`. Given a tolerance factor `p/q` (positive integers with `p >= q`), a pair `{i, j}` is *balanced* when `max(g[i],g[j]) / min(g[i],g[j]) <= p/q`. Count the unordered balanced pairs. Read `n p q` then the `n` gains from stdin; print the count. Constraints: `n <= 2*10^5`, and `g[i], p, q <= 4*10^9` (so values exceed 32-bit, and a product of two of them exceeds 64-bit).

**Key idea — sort + two pointers with a cross-multiplied test.** Never divide. Since all quantities are positive, `g_hi / g_lo <= p/q` is exactly `g_hi * q <= g_lo * p`. Sort the gains ascending and fix the larger member of a pair at index `j`; any partner `i < j` has `g[i] <= g[j]`, so the pair is balanced iff `g[j]*q <= g[i]*p`. For fixed `j` the right side `g[i]*p` increases with `i`, so the balanced partners are a suffix `[lo_j, j-1]`; and as `j` advances `g[j]*q` grows, so the boundary `lo_j` only moves rightward. A single trailing pointer sweeps it: `O(n log n)` total (dominated by the sort). On the sample `g=[10,1,4,8,13,5]`, `p=5`, `q=2` the sorted sweep yields per-`j` partner counts `0,0,1,2,3,2`, summing to `8`.

**Pitfalls.**
1. *Floating point.* `g_hi/(double)g_lo <= p/(double)q` rounds at 53 bits and misjudges pairs whose ratio sits on or near the bound (e.g. `{4,10}` with ratio exactly `5/2`). Use the integer cross-multiplication, not division.
2. *64-bit product overflow.* With values and `p,q` up to `4*10^9`, `g[j]*q` reaches `~1.6*10^19`, past the signed 64-bit ceiling `~9.22*10^18`. Forming the products in `long long` wraps them negative and silently flips the verdict. Concretely, `p=2590928623, q=200071089, g=[3687093964,3758591649]` is a balanced pair (`g_hi*q = 7.52*10^17 <= g_lo*p = 9.55*10^18`), but in `long long` the right product wraps to `~-8.89*10^18`, so the comparison lies and the pair is dropped (`0` instead of `1`). Form both products in `__int128`.
3. *Count overflow.* The answer reaches `n*(n-1)/2 ~ 2*10^10` (all-equal gains, `p>=q`), past 32-bit; the accumulator must be `long long`.
4. *Reading into `int`.* `g[i], p, q` can exceed `2^31`; read them into `long long`.

**Edge cases.** `n = 0` and `n = 1` give `0` (no pairs). All-equal gains with `p >= q`: every pair balanced (`q <= p`), exercising the 64-bit count. `p = q` with distinct gains: only equal-gain pairs would qualify, so `0`. On-threshold pairs (ratio exactly `p/q`) are counted via `<=` with exact integer products. Maximal-magnitude pairs near `4*10^9` exercise the `__int128` path.

**Complexity.** `O(n log n)` time (sort) and `O(n)` space; the sweep is `O(n)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long p, q;
    if (!(cin >> n >> p >> q)) return 0;
    vector<long long> g(n);
    for (auto &x : g) cin >> x;

    // A pair {i, j} (i != j), say with g_lo <= g_hi, is "balanced" when
    //   g_hi / g_lo <= p / q   <=>   g_hi * q <= g_lo * p   (q, p, values all > 0).
    // Sort ascending; for each right endpoint j the admissible left endpoints i (with i < j and
    // g[j]*q <= g[i]*p) form a suffix [lo, j-1] of the sorted prefix, and lo only moves rightward
    // as j advances -> a single two-pointer sweep. With values and p, q up to 4*10^9 the
    // cross-products reach ~1.6*10^19, which OVERFLOWS signed 64-bit; we form them in __int128 so no
    // division or floating point is ever used.
    sort(g.begin(), g.end());

    long long count = 0;     // up to n*(n-1)/2 ~ 2*10^10 at n = 2*10^5, must be 64-bit
    int lo = 0;
    for (int j = 0; j < n; j++) {
        // advance lo to the smallest i with g[j]*q <= g[i]*p, i.e. the pair (i, j) is balanced.
        while (lo < j && (__int128)g[j] * q > (__int128)g[lo] * p) lo++;
        // now i in [lo, j-1] are exactly the partners that pair with j as the larger element.
        count += (long long)(j - lo);
    }

    cout << count << "\n";
    return 0;
}
```

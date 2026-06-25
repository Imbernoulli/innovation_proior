**Problem.** Given `n` integer frequencies `f[0..n-1]` and an inclusive band `[L, R]`, count the
**unordered** pairs `{i, j}` (`i != j`, each pair once) with `L <= |f[i] - f[j]| <= R`. Output the
count modulo `1 000 000 007`. Constraints: `n <= 10^6`, `|f[i]| <= 10^9`, `0 <= L <= R <= 2*10^9`, so
the count can reach `~5*10^11` and the window bounds can be as low as `-3*10^9`.

**Key idea — sort, then a two-pointer prefix sweep.** The gap `|f[i]-f[j]|` is order-independent, so
sorting the frequencies changes nothing. After sorting (`f[0] <= ... <= f[n-1]`), a pair at sorted
positions `i < j` has gap `f[j] - f[i]`, compatible iff `f[j] - R <= f[i] <= f[j] - L`. Fix the
**larger** element `j`; its partners are the prefix indices `i < j` whose value lies in the closed
window `[f[j]-R, f[j]-L]`. Summing these prefix counts over all `j` counts each unordered pair exactly
once — at its larger sorted endpoint. Since `f[j]` is nondecreasing, both window ends are
nondecreasing, so two monotone pointers `lo` (count of prefix values `< f[j]-R`) and `hi` (count of
prefix values `<= f[j]-L`) advance once each; partners `= hi - lo`.

**Pitfalls.**
1. *Double-counting (the headline trap).* The natural "for each fork, count compatible neighbours on
   both the lower and upper side" sums over both endpoints of every pair and yields exactly `2x` the
   answer (on the sample, `16` instead of `8`). Patching with `/2` is fragile under a modulus
   (it invites reducing before halving, destroying the parity, and hides a needless modular inverse).
   The clean fix is structural: count each pair once, at its larger sorted endpoint, over the prefix
   `i < j`.
2. *Inclusive-endpoint asymmetry.* The band is closed, so the upper boundary uses `<=` (`hi` includes
   value `f[j]-L`, the gap-exactly-`L` partner) and the lower boundary uses `<` (`lo` keeps value
   `f[j]-R`, the gap-exactly-`R` partner). Mixing these mis-counts the boundary partners.
3. *Overflow / types.* `f[j]-R` can be `-3*10^9` — outside `int`. Use `long long` for frequencies,
   bounds, and the accumulator. The exact count fits in `long long` (`< 9.2*10^18`); reduce modulo the
   prime only once, at output, so there is no subtraction or division under the modulus.

**Edge cases.** `n = 0` and `n = 1` -> `0` (loop adds nothing). `L = 0` -> equal-frequency forks pair
(the `<=` upper boundary includes value `f[j]`). `R` huge / never binding -> `lo` stays `0`, lower
bound alone governs. `L = 0` with large `R` -> all `C(n, 2)` pairs, matching `n*(n-1)/2 mod p` even at
`n = 10^6`. Negative frequencies are handled identically since only differences matter.

**Complexity.** `O(n log n)` time (dominated by the sort; the sweep is `O(n)`), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    long long L, R;
    cin >> L >> R;
    vector<long long> f(n);
    for (auto &x : f) cin >> x;

    const long long MOD = 1000000007LL;

    sort(f.begin(), f.end());

    // For an unordered pair {i, j} with sorted values f[i] <= f[j], the gap is
    // f[j] - f[i], and it is "compatible" iff L <= f[j] - f[i] <= R.
    // Fix the LARGER element j (right end). Its valid partners are the indices
    // i < j with  f[j] - R <= f[i] <= f[j] - L.
    // As j increases, both window bounds (f[j]-R and f[j]-L) are nondecreasing,
    // so two pointers lo, hi sweep the prefix [0, j) once. Counting only
    // partners to the LEFT of j makes each unordered pair counted exactly once.
    long long total = 0;       // exact count, fits in long long (<= n*(n-1)/2)
    int lo = 0, hi = 0;        // lo: first i with f[i] >= f[j]-R; hi: first i with f[i] > f[j]-L
    for (int j = 0; j < n; j++) {
        // hi: number of elements among the prefix with value <= f[j]-L.
        while (hi < j && f[hi] <= f[j] - L) hi++;
        // lo: number of elements among the prefix with value <  f[j]-R.
        while (lo < j && f[lo] <  f[j] - R) lo++;
        // valid partners i in [lo, hi): values in [f[j]-R, f[j]-L].
        total += (long long)(hi - lo);
    }

    cout << (total % MOD) << "\n";
    return 0;
}
```

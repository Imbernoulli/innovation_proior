**Problem.** Over `n` rounds, each round records an integer score in `0..m`. A round is a *peak* when its score equals `m`. No two consecutive rounds may both be peaks. Count the length-`n` score sequences obeying this rule, modulo `1000000007`. Read `n` and `m` from stdin (`0 <= n <= 2*10^5`, `0 <= m <= 10^9`), print one integer.

**Key idea — two-state linear DP.** The only history the future needs is whether the round just placed was a peak. Carry two counts over the prefix:

- `lo` = valid sequences whose last round is a **non-peak** (one of `m` scores `0..m-1`);
- `hi` = valid sequences whose last round is a **peak** (the single score `m`).

A non-peak may follow anything, a peak may follow only a non-peak, giving (both reading the previous pair):

- `lo_i = (lo_{i-1} + hi_{i-1}) * m`
- `hi_i = lo_{i-1}`

Base case `i = 1`: `lo = m`, `hi = 1`. Answer for `n >= 1` is `(lo + hi) mod p`; for `n = 0` it is `1` (the empty sequence). `O(n)` time, `O(1)` space.

**Why it's correct (numeric self-check).** Sample `n=3, m=2`: `(2,1) -> ((3)*2, 2)=(6,2) -> ((8)*2, 6)=(16,6)`, total `22`. Match. Algebraically for `n=2`: recurrence gives `(m+1)*m + m = m^2 + 2m`, and a direct count gives `(m+1)^2 - 1 = m^2 + 2m`. Identical for all `m`, so the transitions are right.

**Pitfalls.**
1. *32-bit overflow (the headline).* After the modulus, counts reach `~10^9`, and the multiplier `m` reaches `10^9`, so `(lo+hi)*m` is `~10^18`. In `int` this wraps modulo `2^32` *before* `% MOD` runs, so the modulo cannot rescue it. The bug is invisible on the small sample (it prints the correct `22`) but corrupts large inputs: an `int` build returns `513381376` for `n=2, m=10^9` (correct is `35`), and even prints a **negative** `-732834297` for `n=2*10^5, m=10^9` — a negative count is the unmistakable signature of signed overflow. Do every sum, product, and modulo in `long long`, keeping operands `< MOD` so each product stays `< 1.0*10^18`, safely inside 64-bit range.
2. *Reduce before multiply.* Take `% MOD` on the sum and on `m` before forming the product, not after.

**Edge cases.** `n = 0` -> `1` (special-cased before the loop). `n = 1` -> `m + 1` (loop body never runs). `m = 0` -> every round is forced to score `0`, which is itself a peak, so `lo[1]=0, hi[1]=1` and for `n >= 2` the count collapses to `0`.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    const long long MOD = 1000000007LL;
    long long n, m;
    if (!(cin >> n >> m)) return 0;

    // lo = number of valid length-i sequences whose last round is a non-peak
    //      score (one of the m values 0..m-1);
    // hi = number whose last round is the peak score m (the single value m).
    // A peak may not immediately follow a peak.
    //
    // For i = 1: lo = m (scores 0..m-1), hi = 1 (the peak).
    // Transition for one more round:
    //   new_lo = (lo + hi) * m   (any previous ending may be followed by a
    //                             non-peak, and there are m non-peak scores)
    //   new_hi = lo              (a peak may only follow a non-peak; 1 score)
    //
    // n can be 0 (empty sequence: exactly one, the empty one).
    long long lo = (m % MOD), hi = 1 % MOD;
    if (n == 0) { cout << 1 % MOD << "\n"; return 0; }

    for (long long i = 2; i <= n; i++) {
        long long nlo = ((lo + hi) % MOD) * (m % MOD) % MOD;
        long long nhi = lo;
        lo = nlo;
        hi = nhi;
    }

    cout << (lo + hi) % MOD << "\n";
    return 0;
}
```

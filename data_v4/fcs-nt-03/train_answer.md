**Problem.** Given a single integer `n` (`0 <= n <= 10^12`), compute the divisor summatory function
`D(n) = d(1) + d(2) + ... + d(n)`, where `d(k)` is the number of positive divisors of `k`. Equivalently
`D(n) = sum_{i=1}^{n} floor(n/i)`. Read `n` from stdin, print `D(n)`. By convention `D(n) = 0` for `n <= 0`.

**Why the obvious approaches are too slow.** The dividend view `D(n) = sum_k d(k)` via a divisor sieve is
`O(n log n)` time and `O(n)` memory — at `n = 10^12` that is terabytes of RAM and tens of trillions of
operations, hopeless on both counts. Rewriting as `D(n) = sum_{i=1}^{n} floor(n/i)` (each `i` divides
`floor(n/i)` of the numbers in `1..n`) removes the memory problem but is still an `O(n)` loop — a trillion
iterations, far over a one-second limit. The linear barrier must be broken, not micro-optimized.

**Key idea — the Dirichlet hyperbola method.** `D(n)` equals the number of lattice points `(a, b)` with
`a, b >= 1` and `a * b <= n` (each `k <= n` contributes one point per divisor pair `k = a*b`, hence `d(k)`
points). That region is symmetric about `a = b`. Let `s = floor(sqrt(n))`. The vertical strip `a <= s` under
the hyperbola contains `T = sum_{a=1}^{s} floor(n/a)` points; by `a <-> b` symmetry the horizontal strip
`b <= s` contains the same `T`. Their union is the whole region, and their overlap is exactly the `s × s`
square (`a, b <= s` implies `a*b <= s*s <= n`). Inclusion–exclusion gives

```
D(n) = 2 * T - s*s,    T = sum_{i=1}^{s} floor(n/i),    s = floor(sqrt(n)).
```

This is `O(sqrt(n))` time and `O(1)` memory — about `2*10^6` iterations at `n = 10^12`, sub-millisecond.
(The same `O(sqrt(n))` could be reached by block-decomposing the constant runs of `floor(n/i)`, but the
hyperbola identity is cleaner and array-free.)

**Pitfalls.**
1. *Integer square root.* `s = floor(sqrt(n))` via floating `sqrtl` can round either up or down near
   `10^12`. If `s` is one too large, the `s × s` square pokes outside the hyperbola and the correction term
   is wrong. Clamp `s` to the exact floor with two `O(1)` loops: `while (s*s > n) s--;` then
   `while ((s+1)*(s+1) <= n) s++;`. The straddlers `48, 49, 50` are the sharpest test of this.
2. *Overflow.* `D(10^12) ≈ 2.8*10^13` and `2*T ≈ 5.7*10^13`; all of it needs `long long`. An `int` is a
   silent wrong-answer. The guard product `(s+1)^2 ≈ 10^12` also fits in 64 bits.

**Edge cases (all handled).** `n = 0` -> `0` (early return for `n <= 0`); `n = 1` -> `1`; `n = 2` -> `3`;
perfect squares `n = s*s` (correction term exact thanks to the clamp); empty input -> print nothing.

**Complexity.** `O(sqrt(n))` time, `O(1)` extra space.

**Verification.** Differential-tested against an independent `O(n log n)` divisor sieve on `1028` cases
(explicit edges + perfect squares + square-straddlers + 1000 random `n` up to `2*10^4`): zero mismatches.
Cross-checked at `n = 10^12` against an exact-`isqrt` Python hyperbola (`27785452449086`); runs in
sub-millisecond with a few MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n;
    if (!(cin >> n)) return 0;          // no input -> nothing to do
    if (n <= 0) { cout << 0 << "\n"; return 0; } // D(n)=0 for n<=0 by convention

    // Dirichlet hyperbola method for D(n) = sum_{i=1..n} d(i) = sum_{i=1..n} floor(n/i).
    // Pairs (a,b) with a*b <= n are counted by symmetry across the line a=b:
    //   D(n) = 2 * sum_{i=1..s} floor(n/i) - s*s,  where s = floor(sqrt(n)).
    long long s = (long long)sqrtl((long double)n);
    while (s * s > n) s--;               // guard sqrt rounding from above
    while ((s + 1) * (s + 1) <= n) s++;  // guard sqrt rounding from below

    long long sum = 0;
    for (long long i = 1; i <= s; i++) sum += n / i;

    long long answer = 2 * sum - s * s;
    cout << answer << "\n";
    return 0;
}
```

**Problem.** Given `n` integers `a[0..n-1]` (each may be positive, negative, or zero) and a modulus `m`, count the **non-empty** subsequences whose **product is strictly positive**, and print that count modulo `m` (in `[0, m-1]`). Read `n`, `m`, then the `n` values from stdin.

**Key idea — only signs matter, so it is a parity count.** A subset's product is strictly positive iff it contains **no zero** and an **even number of negatives** (zero negatives included). Let `P = #positives`, `N = #negatives`; zeros are unusable. Positives are free (`2^P` ways); negatives must form an even-sized subset. The number of even-sized subsets of an `N`-set is

- `E_N = 2^(N-1)` for `N >= 1` (the binomial identity `sum_{k even} C(N,k) = 2^(N-1)`), and
- `E_N = 1` for `N = 0` (only the empty, even-sized, choice).

So the positive-product subsets *including the empty one* number `2^P * E_N`, and the answer is `(2^P * E_N - 1) mod m`, the `-1` dropping the empty subset. All powers are computed with modular fast exponentiation.

**Pitfalls.**
1. *Base case of the even-subset count.* `2^(N-1)` is only valid for `N >= 1`. Writing `power_mod(2, N-1, m)` when `N = 0` passes a negative exponent; a guarded loop returns `1` only by accident. Special-case `N == 0` to the deliberate `E_0 = 1`. Getting this wrong corrupts every all-positive / positives-and-zeros input.
2. *Sign handling in the modular subtraction.* `total - 1` can be negative, and C++ `%` keeps the dividend's sign, so `(total - 1) % m` can print a negative value. Use `(total - 1 % m + m) % m`.
3. *`m = 1`.* Initialize the exponentiation accumulator as `1 % mod` (not `1`) and subtract `1 % m`, so every result collapses cleanly to `0`.
4. *Overflow.* `2^P` overflows 64-bit before reduction; reduce with modular exponentiation and use `__int128` for the products (`(m-1)^2 ≈ 10^18`).

**Edge cases.** Empty array `n = 0` -> `0`. All-negative `[-2,-3,-4]` -> `3` (the three negative pairs), **not** `0`. All-zero `[0,0,0]` -> `0`. Single negative `[-5]` -> `0`. `m = 1` -> `0`. All exercised and matched against a brute-force subset oracle.

**Complexity.** `O(n + log P + log N)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count non-empty subsequences whose product is strictly positive, modulo m.
// A subset has a strictly positive product iff it contains NO zero element and an
// EVEN number (0, 2, 4, ...) of negative elements. Positives are unconstrained.
// Let P = #positives, N = #negatives. Zeros can never be part of such a subset.
//   E_N = number of even-sized subsets of the N negatives
//       = 2^(N-1)  if N >= 1   (exactly half of the 2^N subsets are even-sized),
//       = 1        if N == 0   (only the empty choice, which is even-sized).
//   total = 2^P * E_N          (subsets with no zero and even #negatives, empty allowed)
//   answer = (total - 1) mod m (remove the single empty subset), kept non-negative.

long long power_mod(long long base, long long exp, long long mod) {
    base %= mod;
    if (base < 0) base += mod;
    long long result = 1 % mod;            // 1 % mod handles mod == 1
    while (exp > 0) {
        if (exp & 1) result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, m;
    if (!(cin >> n >> m)) return 0;

    long long P = 0, N = 0; // zeros are counted by neither: they can never be chosen
    for (long long i = 0; i < n; i++) {
        long long x;
        cin >> x;
        if (x > 0) P++;
        else if (x < 0) N++;
        // x == 0 contributes to neither P nor N
    }

    long long posWays = power_mod(2, P, m);              // 2^P
    long long evenNeg;
    if (N == 0) evenNeg = 1 % m;                         // base case: NOT 2^(-1)
    else evenNeg = power_mod(2, N - 1, m);               // 2^(N-1)

    long long total = (__int128)posWays * evenNeg % m;   // includes the empty subset, in [0, m)
    long long answer = (total - 1 % m + m) % m;          // remove empty subset, keep non-negative

    cout << answer << "\n";
    return 0;
}
```

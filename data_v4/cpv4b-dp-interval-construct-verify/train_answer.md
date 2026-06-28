**Problem.** Read an integer `n` (`1 <= n <= 2*10^5`). Output `n` integer marks
`0 = x[0] < x[1] < ... < x[n-1] <= M`, with `M = 8*n*n`, such that all `n*(n-1)/2` pairwise
differences `x[j] - x[i]` (`i < j`) are pairwise distinct — a Sidon set / perfect ruler. Special
judge: any placement satisfying the four conditions (starts at 0, strictly increasing, last `<= M`,
distinct differences) is accepted.

**Why the tempting construction is wrong.** Doubling — mark `k` at `2^k` — is a *genuine* Sidon set
(unique binary representations make every subset sum, hence every pairwise difference, distinct), so it
passes the distinctness check on every small input. But its largest mark is `2^(n-1)`, and the budget
is `M = 8*n*n`. The crossover is at `n = 11`: `2^10 = 1024 > 8*121 = 968`. For `n <= 10` doubling fits
(`2^9 = 512 <= 800`); from `n = 11` it overflows the rail, exponentially worse as `n` grows. A tester
who checks only "are the differences distinct?" on `n <= 10` sees a flawless run, ships it, and scores
**zero** at `n = 10^5` — not because differences collide, but because the ruler is astronomically too
long. Correct on the property, wrong on the budget.

**Key idea — the Erdős–Turán Sidon set.** For a prime `p`, the set
`b[k] = 2*p*k + (k^2 mod p)`, `k = 0..p-1`, is a Sidon set with all elements in `[0, ~2*p^2)`. To hit
target size `n`, pick the smallest prime `p >= n` (one exists in `[n, 2n)` by Bertrand) and take the
first `n` elements; a subset of a Sidon set is still Sidon, and the sequence is strictly increasing in
`k` (`b[k+1]-b[k] = 2p + (r(k+1)-r(k)) >= p+1 > 0`). Distinctness: if `b[k1]-b[k2] = b[k3]-b[k4]`, the
`2p`-scale and the `<p` residue parts separate, forcing `k1-k2 = k3-k4 = d` and
`d*(k1+k2) ≡ d*(k3+k4) (mod p)`; with `p` *prime* and `0 < |d| < p`, `d` is invertible, so
`k1+k2 ≡ k3+k4`, pinning `{k1,k2} = {k3,k4}`. Budget: largest mark `< 2*p*n < 2*(2n)*n = 4*n^2 <= 8*n*n`,
fitting `M` with a 2x margin.

**Pitfalls.**
1. *Budget vs. property.* Passing distinctness on small `n` says nothing about the length budget at
   large `n`. Doubling proves this: right property, ruined budget. Always verify the constraint that
   *scales* (here `x[n-1] <= 8 n^2`), not just the one that is easy to eyeball.
2. *Primality is essential.* The shortcut `p = n` (skip the prime search) collides the moment `n` is
   composite: for `n = 4`, `b = [0, 9, 16, 25]` has `16-0 = 25-9 = 16`. It only survives when `n` is
   prime. Use the next *actual* prime.
3. *Overflow.* With `n, p ~ 2*10^5`, the products `2*p*k` and `k*k` reach `~8*10^10` and `~4*10^10`,
   overflowing 32-bit. Make `p` a `long long` and cast `k` so every multiply is 64-bit; coordinates
   reach `~8*10^10` and must be printed as 64-bit.

**Edge cases.** `n = 1`: print `0` (single mark, no pair, vacuously valid). `n = 2`: `[0, 5]`. The
strictly-increasing requirement is automatic from the `2p`-dominated step; the first mark is forced to
`0` by subtracting `base = b[0]` defensively.

**Complexity.** `O(n)` to build the set plus an `O(sqrt(p))`-per-candidate prime search over `O(n)`
candidates in the worst gap — far under the limits; effectively near-linear, tens of milliseconds at
`n = 2*10^5`. `O(n)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Smallest prime p with p >= n (n >= 1). Trial division is fine: the prime we need is < 2n,
// and we only test odd candidates up to sqrt(candidate).
static bool isPrime(long long v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (long long d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}
static long long nextPrimeAtLeast(long long n) {
    long long p = max(n, 2LL);
    while (!isPrime(p)) p++;
    return p;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // n marks 0 = x_0 < x_1 < ... < x_{n-1} <= M with all pairwise differences distinct (Sidon set),
    // M = 8*n*n. Erdos-Turan: with a prime p, b[k] = 2*p*k + (k^2 mod p) (k = 0..p-1) is a Sidon set.
    // Take the first n elements (a subset of a Sidon set is Sidon) and shift so the smallest is 0.

    if (n == 1) {            // single mark sits at the origin; no pair, vacuously distinct
        cout << "0\n";
        return 0;
    }

    long long p = nextPrimeAtLeast(n);          // p in [n, 2n) by Bertrand's postulate
    vector<long long> b(n);
    for (int k = 0; k < n; k++)
        b[k] = 2 * p * (long long)k + ((long long)k * k) % p;   // already strictly increasing in k
    long long base = b[0];                       // b[0] = 0 here, but subtract defensively
    for (int k = 0; k < n; k++) b[k] -= base;

    // b is strictly increasing, so it is already sorted; emit as the mark coordinates.
    for (int k = 0; k < n; k++) {
        cout << b[k];
        cout << (k + 1 == n ? '\n' : ' ');
    }
    return 0;
}
```

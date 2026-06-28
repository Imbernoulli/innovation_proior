**Problem.** Given `n` (`2 <= n <= 3000`), output `n` distinct integers in `[0, 4*n*n]` that form a **Sidon set**: all `C(n, 2)` positive pairwise differences are pairwise distinct (equivalently all unordered pairwise sums are distinct). Any valid set is accepted. Read `n` from stdin; print the values space-separated on one line.

**Key idea — Erdős–Turán, certified at scale.** For a prime `p`, the set

```
a_k = 2*p*k + (k*k mod p),   k = 0 .. n-1
```

is a Sidon set contained in `[0, 2*p*p)`. Choose the smallest prime `p >= n` and take the first `n` elements. Why Sidon: write `a_k = 2*p*k + r_k` with `r_k = k*k mod p` in `[0, p)`. If `a_j - a_i = a_l - a_m`, the `2p`-multiple parts and the residual parts (each in `(-p, p)`) must match separately, giving `j - i = l - m =: d` and `d*(i+j) ≡ d*(l+m) (mod p)`; since `p` is prime and `0 < d < p`, `d` is invertible, so `i + j ≡ m + l`, which (with indices in a width-`< p` window) forces `i = m, j = l`. The largest element is at `k = n-1`: `2*p*(n-1) + ((n-1)^2 mod p) < 2*p*n`. The smallest prime `>= n` satisfies `p <= 1.375*n` over the whole input range, so `max < 2*1.375*n*n < 4*n*n` — always in band. After building, **certify the actual output**: collect all pairwise differences, sort, and do a single two-pointer adjacency pass to confirm no two neighbours are equal.

**Pitfalls.**
1. *A correct Sidon set that violates the range.* The greedy Mian–Chowla set (sweep `x` upward, keep it if no difference repeats) is always Sidon, so it passes any *property* check on small `n`. But its `n`-th value grows like `~n^3` while the band grows like `4n^2`; they cross near `n ≈ 175`. It stays in band through `n = 150` and overflows from `n ≈ 200` (e.g. `n = 200`: greedy max `172921 > 160000`). Validating only on `n <= 10` looks perfect and scores `0` on the large tests. You must check the property *and the range* at the real `n`.
2. *Wrong modulus.* Using `n` as the modulus instead of a prime breaks the construction whenever `n` is composite: residues `k*k mod n` repeat and `d` is no longer invertible. For `n = 4` the set `{0, 9, 16, 25}` has the difference `16` twice. The modulus must be the smallest prime `p >= n`.
3. *Two-pointer off-by-one.* In the duplicate scan, gate the loop on the trailing pointer (`hi < size`), not `lo`, or you read one past the end.

**Edge cases.** `n = 2` → `{0, 5}` (`p = 2`); `n = 3` → `{0, 7, 13}` (`p = 3`); `n = 5` → `{0, 11, 24, 34, 41}` (`p = 5`, the documented sample). Strict increase is automatic since `a_{k+1} - a_k = 2p + (r_{k+1} - r_k) > 0`. At `n = 3000`, `p = 3001` and `max ≈ 1.8*10^7 < 3.6*10^7`.

**Complexity.** Construction `O(n)` (plus a cheap prime search). The Sidon certification is `O(n^2 log n)` over `~4.5*10^6` differences for the largest `n` — about `36 MB` and well under `2 s`. Without the certify, output and construction alone are `O(n)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Smallest prime >= x (x small here: x <= ~3000, so trial division is fine).
static bool isPrime(long long x) {
    if (x < 2) return false;
    if (x % 2 == 0) return x == 2;
    for (long long d = 3; d * d <= x; d += 2)
        if (x % d == 0) return false;
    return true;
}
static long long smallestPrimeGE(long long x) {
    long long p = max(2LL, x);
    while (!isPrime(p)) p++;
    return p;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // Erdos-Turan B2 (Sidon) set: pick a prime p >= n, then
    //   a_k = 2*p*k + (k*k mod p),   k = 0 .. n-1.
    // The full set k=0..p-1 has all pairwise differences distinct; any subset
    // (here the first n) inherits that, so the n elements form a Sidon set.
    // Max element is at k=n-1: 2*p*(n-1)+((n-1)^2 mod p) < 2*p*n <= 2*(1.4n)*n < 4n^2.
    long long p = smallestPrimeGE(n);

    vector<long long> a(n);
    for (int k = 0; k < n; k++) {
        long long km = (long long)k % p;
        a[k] = 2 * p * (long long)k + (km * km) % p;
    }

    // ---- Two-pointer certification at the REQUIRED scale (not just tiny cases) ----
    // a[] is strictly increasing. Collect all positive pairwise differences,
    // sort them, then a single two-pointer (adjacent) pass detects any duplicate.
    // If a duplicate exists the set is NOT Sidon. This certifies the construction
    // for the actual n, instead of trusting it from small hand examples.
    {
        // strictly increasing check
        for (int i = 1; i < n; i++) assert(a[i] > a[i - 1]);
        // range check
        long long L = 4LL * n * n;
        for (int i = 0; i < n; i++) assert(a[i] >= 0 && a[i] <= L);
        // all pairwise differences distinct (Sidon) via sort + two-pointer
        vector<long long> diffs;
        diffs.reserve((size_t)n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                diffs.push_back(a[j] - a[i]);
        sort(diffs.begin(), diffs.end());
        for (size_t lo = 0, hi = 1; hi < diffs.size(); lo++, hi++)
            assert(diffs[lo] != diffs[hi]);
    }

    // Output the constructed set, space-separated on one line.
    string out;
    out.reserve((size_t)n * 8);
    for (int i = 0; i < n; i++) {
        if (i) out.push_back(' ');
        out += to_string(a[i]);
    }
    out.push_back('\n');
    fputs(out.c_str(), stdout);
    return 0;
}
```

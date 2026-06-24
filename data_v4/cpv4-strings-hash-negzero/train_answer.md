**Problem.** Given `a[0..n-1]` (values may be negative or zero, `|a[i]| <= 10^9`, `n <= 2*10^5`), find the length of the longest *block* (contiguous subarray) that occurs at two or more **distinct** starting positions; overlaps are allowed. Output `0` if no block of length `>= 1` repeats. Read `n` and the values from stdin, print the length.

**Key idea — binary search the length + rolling hash.** The predicate `hasDup(L)` = "some length-`L` block repeats" is monotone: if a length-`L` block sits at distinct starts `i != j`, dropping the last element leaves a length-`(L-1)` block at the same distinct starts, so `hasDup(L) => hasDup(L-1)`. Binary-search the largest `L` in `[1, n]` with `hasDup(L)` true. Each query answers "do two length-`L` windows match?" in `O(n)` with a polynomial rolling hash: precompute prefix hashes so a window's fingerprint is `pre[r] - pre[l]*B^{r-l}` mod `p`, scan all `n-L+1` windows, and report a repeat when two fingerprints collide. Overall `O(n log n)`.

**Pitfalls.**
1. *The numeric alphabet (sign/zero).* A textbook string hash assumes each symbol is a positive digit. Feeding a raw `(u64)v` for a negative `v` casts it to `2^64 + v`, which folds the value space onto itself modulo `p = 2^61 - 1`: since `2^64 ≡ 8 (mod p)`, the values `-3` and `5` get the **same** fingerprint, a false-equal that inflates `hasDup`. Cure: offset every symbol into a strictly positive, injective range — `v -> v + 10^9 + 1` lands in `[1, 2*10^9+1]`, far below `p`, with no symbol equal to `0`. Now `0` and negatives are ordinary digits.
2. *The base case of the search.* Starting the binary search at `lo = 0` lets `mid` reach `hasDup(0)`, whose empty-window fingerprints all collide into a vacuous `true` (and `windowEqual` with `L=0` is vacuously true). Start at `lo = 1`, and guard the predicate with `L <= 0 => true` (never queried) and `L > n => false`.
3. *Exactness.* Confirm every fingerprint match with a direct `windowEqual` comparison, and run two independent 61-bit hashes, so a hash clash can never produce a wrong answer.

**Edge cases.** `n = 0` and the no-token input print `0` and return early (no array, no division by a base range, no loop). `n = 1` -> `0`. All-negative with no repeat -> `0`; with a repeated value -> `1`. All zeros (`n` of them) -> `n-1` (an overlapping block of zeros). Large magnitudes near `+-10^9` stay injective after the offset and never overflow because `mulmod` uses `__int128`.

**Complexity.** `O(n log n)` time (`O(log n)` queries, each a linear scan), `O(n)` memory for prefix hashes and the per-query hash table.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __int128 i128;

static int n;
static vector<long long> a;

// Two independent 61-bit Mersenne-prime hashes to make collisions astronomically rare.
static const u64 MOD = (1ULL << 61) - 1;

static inline u64 mulmod(u64 x, u64 y) {
    i128 z = (i128)x * y;
    u64 lo = (u64)(z & MOD);
    u64 hi = (u64)(z >> 61);
    u64 r = lo + hi;
    if (r >= MOD) r -= MOD;
    return r;
}
static inline u64 addmod(u64 x, u64 y) {
    u64 r = x + y;
    if (r >= MOD) r -= MOD;
    return r;
}

static u64 BASE1, BASE2;
static vector<u64> pre1, pre2;   // prefix hashes
static vector<u64> pw1, pw2;     // base powers

// hashed symbol for a value: offset so negatives/zeros become positive and never 0.
// |a[i]| <= 1e9, so a[i] + OFFSET is in [1, ~2e9], strictly positive => no symbol hashes to 0.
static const u64 OFFSET = 1000000001ULL;

static inline u64 sym(long long v) {
    return (u64)(v + (long long)OFFSET); // in [1, 2000000001], always > 0
}

// hash of subarray a[l .. l+L-1] (0-indexed), combined 128-bit-ish key.
static inline pair<u64,u64> windowHash(int l, int L) {
    int r = l + L; // exclusive
    // h = pre[r] - pre[l]*pw[L]
    u64 h1 = (pre1[r] + (MOD - mulmod(pre1[l], pw1[L]))) % MOD;
    u64 h2 = (pre2[r] + (MOD - mulmod(pre2[l], pw2[L]))) % MOD;
    return {h1, h2};
}

static inline bool windowEqual(int i, int j, int L) {
    for (int k = 0; k < L; k++) if (a[i + k] != a[j + k]) return false;
    return true;
}

// Does some block of length L occur at >=2 distinct start positions (overlap allowed)?
static bool hasDup(int L) {
    if (L <= 0) return true;          // empty block trivially "repeats"; not queried for L>=1
    if (L > n) return false;
    // map combined hash key -> list of start indices sharing that key; verify on clash.
    unordered_map<u64, vector<int>> seen;
    seen.reserve((size_t)(n - L + 1) * 2 + 4);
    for (int i = 0; i + L <= n; i++) {
        auto h = windowHash(i, L);
        u64 key = h.first * 1000000007ULL + h.second; // mix two 61-bit hashes
        auto &bucket = seen[key];
        for (int j : bucket) if (windowEqual(j, i, L)) return true; // exact confirmation
        bucket.push_back(i);
    }
    return false;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) { cout << 0 << "\n"; return 0; }
    a.resize(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // random bases in a safe range [256, MOD-2]
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    BASE1 = 256 + rng() % (MOD - 300);
    BASE2 = 256 + rng() % (MOD - 300);

    pre1.assign(n + 1, 0);
    pre2.assign(n + 1, 0);
    pw1.assign(n + 1, 1);
    pw2.assign(n + 1, 1);
    for (int i = 0; i < n; i++) {
        pre1[i + 1] = addmod(mulmod(pre1[i], BASE1), sym(a[i]) % MOD);
        pre2[i + 1] = addmod(mulmod(pre2[i], BASE2), sym(a[i]) % MOD);
        pw1[i + 1] = mulmod(pw1[i], BASE1);
        pw2[i + 1] = mulmod(pw2[i], BASE2);
    }

    // Binary search the largest L in [1, n] with hasDup(L) true.
    // Monotone: if a block of length L repeats, its length-(L-1) prefixes also repeat.
    int lo = 1, hi = n, ans = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (hasDup(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    cout << ans << "\n";
    return 0;
}
```

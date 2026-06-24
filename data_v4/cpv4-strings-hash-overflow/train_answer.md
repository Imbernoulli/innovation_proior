**Problem.** Given a string `s` of length `n` and a window length `L`, slide a length-`L` window over `s` to get `n - L + 1` windows. Group windows by identical content; for a content occurring `c` times it contributes `C(c, 2) = c*(c-1)/2` pairs. The **echo score** is the sum of these contributions over all distinct contents. Read `n`, `L`, and `s` from stdin and print the echo score. If `L > n` the score is `0`.

**Why the obvious method is too slow.** Materializing all windows as substrings and sorting them is `O(n L log n)` time and `O(n L)` memory; with `L` up to `2*10^5` that is tens of gigabytes and `~10^{10}` character comparisons. It is correct (it is the definition) and makes a fine brute force, but it cannot be the shipped solution.

**Key idea — polynomial hashing.** Fingerprint each window in `O(1)` so grouping no longer depends on `L`. With prefix hashes `h[i+1] = h[i]*B + val(s[i])` and powers `p[i] = B^i` (all mod `M`), the hash of the window `[l, l+L)` is

```
hash = (h[l + L] - h[l] * p[L]) mod M     // add M before the mod to stay non-negative
```

Use a **double hash** (two independent moduli/bases) and treat two windows as equal only when both fingerprints match; the effective modulus `~10^{18}` drops the expected number of accidental cross-content collisions among `2*10^5` windows to `~10^{-8}`. Offset characters by `+1` (so the smallest character is not value `0`) to avoid `"a"`, `"aa"`, `"aaa"` all hashing to `0`. Pack the two residues into one 64-bit key, sort the `m = n - L + 1` keys, count run lengths, and add `C(c, 2)` per run.

**Correctness.** Equal content implies equal fingerprint always; the double modulus makes the converse hold with overwhelming probability, so runs of equal keys are exactly the content groups. `n - L + 1` covers every window start `0 .. n - L`. Summing `C(c, 2)` over groups is the definition of the echo score. Validated against an independent substring-dictionary brute force on 600 random small cases with zero mismatches.

**Pitfalls.**
1. *Int overflow at the multiplication (the headline trap).* A single content can occur `~n` times, so `C(c, 2) ~ n^2/2 ~ 2*10^{10}` and the total exceeds the 32-bit range. A `long long` *accumulator* is not enough: if `c` is `int`, the product `c*(c-1)` is computed in 32-bit and wraps **before** being widened — `c = 200000` gives a true `39,999,800,000` but the `int` result is `1,345,094,336`, so the answer comes out `672,547,168` instead of `19,999,900,000`. Make `c` a `long long` so the multiplication itself is 64-bit. Small tests never expose this; only `n` near `2*10^5` does.
2. *Off-by-one in the window count.* The number of length-`L` windows is `n - L + 1`, not `n - L`; the latter drops the last window and undercounts a group.
3. *Single-modulus collisions.* One `~10^9` modulus yields `~10` expected accidental collisions across `2*10^5` windows, merging distinct contents. Use a double hash.

**Edge cases.** `L > n` -> `0` (guard before allocating `keys`, else a negative size becomes a huge `size_t`); `L = n` -> one window, score `0`; `n = 1` -> `0`; all-distinct windows -> `0`; all-identical / highly periodic -> the overflow witness, must stay 64-bit throughout.

**Complexity.** `O(n)` to build prefix hashes and fingerprints, `O(n log n)` to sort the keys, `O(n)` to scan runs; `O(n)` memory, independent of `L`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, L;
    if (!(cin >> n >> L)) return 0;
    string s;
    cin >> s;

    // If L > n there are no windows -> echo score 0.
    if (L > n || L <= 0) {
        cout << 0 << "\n";
        return 0;
    }

    int m = n - L + 1; // number of windows

    // Double polynomial hashing to make collisions astronomically unlikely.
    const unsigned long long MOD1 = 1000000007ULL;
    const unsigned long long MOD2 = 998244353ULL;
    const unsigned long long B1 = 131ULL;
    const unsigned long long B2 = 137ULL;

    // Precompute prefix hashes.
    vector<unsigned long long> h1(n + 1, 0), h2(n + 1, 0), p1(n + 1, 1), p2(n + 1, 1);
    for (int i = 0; i < n; i++) {
        unsigned long long c = (unsigned long long)(unsigned char)s[i] + 1ULL;
        h1[i + 1] = (h1[i] * B1 + c) % MOD1;
        h2[i + 1] = (h2[i] * B2 + c) % MOD2;
        p1[i + 1] = (p1[i] * B1) % MOD1;
        p2[i + 1] = (p2[i] * B2) % MOD2;
    }

    auto getHash = [&](int l) -> unsigned long long {
        // hash of window [l, l+L)
        unsigned long long x1 = (h1[l + L] + MOD1 - (h1[l] * p1[L]) % MOD1) % MOD1;
        unsigned long long x2 = (h2[l + L] + MOD2 - (h2[l] * p2[L]) % MOD2) % MOD2;
        return (x1 << 32) ^ x2; // combine two 32-bit-ish hashes into one key
    };

    // Group windows by combined hash, count multiplicities.
    vector<unsigned long long> keys(m);
    for (int i = 0; i < m; i++) keys[i] = getHash(i);
    sort(keys.begin(), keys.end());

    // Echo score = sum over distinct contents of c*(c-1)/2.
    // c can be ~2e5, so c*(c-1)/2 ~ 2e10 and the total can exceed 32-bit range.
    long long answer = 0;
    int i = 0;
    while (i < m) {
        int j = i;
        while (j < m && keys[j] == keys[i]) j++;
        long long c = j - i;
        answer += c * (c - 1) / 2;
        i = j;
    }

    cout << answer << "\n";
    return 0;
}
```

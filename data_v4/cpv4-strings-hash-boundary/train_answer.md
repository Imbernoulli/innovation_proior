**Problem.** Given a lowercase string `s` and `q` queries, each a 1-indexed **inclusive** range `(l, r)`, output the smallest period of the substring `t = s[l..r]`. A period `p` (`1 <= p <= len`, `len = r-l+1`) satisfies `t[i] == t[i+p]` for all `0 <= i < len-p`, equivalently the length-`(len-p)` prefix equals the length-`(len-p)` suffix. `p = len` is always a period (length-0 overlap), so the answer exists and lies in `[1, len]`. Constraints: `|s|, q <= 5000`.

**Key idea — prefix hashing + smallest-`p` scan.** Build a forward polynomial prefix-hash table so the hash of any inclusive window `s[a..b]` is `O(1)`. Then `p` is a period of `s[L..R]` (with `L=l-1, R=r-1`) iff the two equal-length windows are equal:

- prefix `t[0..len-p-1] = s[L .. R-p]`,
- suffix `t[p..len-1]   = s[L+p .. R]`,

both of length `len - p`. So a period test is one hash equality `get(L, R-p) == get(L+p, R)`. Scan `p = 1, 2, ...` and return the first that matches, else `len`. Each test is `O(1)`, a query is `O(len)`, total `O(|s| + sum of len)` — comfortable at `5000*5000 = 2.5*10^7` `O(1)` tests under 1 s.

**Pitfalls — every boundary is load-bearing.**
1. *Extraction off-by-one.* With an exclusive-top table (`H[i]` = hash of `s[0..i-1]`, `H[0]=0`), the hash of inclusive `s[a..b]` is `H[b+1] - H[a]*pow[b-a+1]`: the upper index is `b+1` (one past the inclusive end), and the power exponent is the **window length** `b-a+1`. Writing `H[b]` hashes a window one char too short — `get(0,0)` then returns the empty-window hash `0` instead of `s[0]`.
2. *Period-window off-by-one.* The suffix starts at `L+p`, **not** `L+p+1`. An over-shift makes the suffix one char shorter than the prefix, so equal substrings stop comparing equal and the true period is silently skipped (e.g. period `2` of `"abab"` gets missed).
3. *Hash collision.* A single modulus can be broken on adversarial input; use a **double** hash (two moduli/bases) so a false equality needs a simultaneous collision (`~10^{-18}`).
4. *Overflow.* Use `unsigned long long` and reduce each product `% MOD` before it can exceed 64 bits.

**Edge cases.** `len = 1` (single-char string or `(k,k)` query): the `p` loop is empty, answer `1`, and no degenerate hash window is formed. All-equal `a^k`: period `1`. Aperiodic like `aab`: falls through to `len`. Non-1 left end: the `L = l-1` conversion keeps the windows aligned. Largest tested `p = len-1` yields two valid length-1 windows; `p = len` is never hashed, only the fallback.

**Complexity.** `O(|s|)` preprocessing, `O(1)` per period test, `O(|s| + sum over queries of len)` overall; `O(|s|)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Smallest period of a queried substring, via double polynomial hashing.

  For a query (l, r) [1-indexed, inclusive], let len = r - l + 1 and t = s[l..r].
  A period p (1 <= p <= len) is valid iff t has period p, i.e. the prefix of t
  of length (len - p) equals the suffix of t of length (len - p):
        t[0 .. len-p-1] == t[p .. len-1].
  In original-string terms (0-indexed internal indices L = l-1, R = r-1):
        s[L .. R-p]  ==  s[L+p .. R],   both of length len - p.
  When p == len, the overlap length is 0, which is vacuously a valid period,
  so a period in [1, len] always exists and the answer is well defined.

  We want the smallest valid p. We just scan p = 1, 2, ... and test each with a
  hash comparison of two equal-length substrings. (The hashing makes each test
  O(1); the scan itself is what the brute force also does, so this stays an
  honest reference for correctness.)
*/

struct Hasher {
    int n;
    unsigned long long MOD1, MOD2, B1, B2;
    vector<unsigned long long> h1, h2, p1, p2;
    Hasher(const string &s, unsigned long long MOD1_, unsigned long long MOD2_,
           unsigned long long B1_, unsigned long long B2_)
        : n((int)s.size()), MOD1(MOD1_), MOD2(MOD2_), B1(B1_), B2(B2_) {
        h1.assign(n + 1, 0);
        h2.assign(n + 1, 0);
        p1.assign(n + 1, 0);
        p2.assign(n + 1, 0);
        p1[0] = 1;
        p2[0] = 1;
        for (int i = 0; i < n; i++) {
            unsigned long long c = (unsigned long long)(s[i] - 'a' + 1);
            // prefix hash on [0, i]: H[i+1] covers s[0..i] (exclusive upper index i+1)
            h1[i + 1] = (h1[i] * B1 + c) % MOD1;
            h2[i + 1] = (h2[i] * B2 + c) % MOD2;
            p1[i + 1] = (p1[i] * B1) % MOD1;
            p2[i + 1] = (p2[i] * B2) % MOD2;
        }
    }
    // hash of s[a .. b] INCLUSIVE, 0-indexed, length = b - a + 1.
    pair<unsigned long long, unsigned long long> get(int a, int b) const {
        int lenq = b - a + 1;
        unsigned long long x1 = (h1[b + 1] + MOD1 - (h1[a] * p1[lenq]) % MOD1) % MOD1;
        unsigned long long x2 = (h2[b + 1] + MOD2 - (h2[a] * p2[lenq]) % MOD2) % MOD2;
        return {x1, x2};
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int q;
    if (!(cin >> q)) return 0;

    Hasher H(s, 1000000007ULL, 998244353ULL, 131ULL, 137ULL);

    string out;
    out.reserve((size_t)q * 7);

    for (int Q = 0; Q < q; Q++) {
        int l, r;
        cin >> l >> r;
        int L = l - 1, R = r - 1;          // 0-indexed inclusive bounds
        int len = R - L + 1;

        int ans = len;                      // p = len is always valid (overlap length 0)
        for (int p = 1; p < len; p++) {
            // compare s[L .. R-p] with s[L+p .. R]; both have length (len - p) >= 1
            if (H.get(L, R - p) == H.get(L + p, R)) {
                ans = p;
                break;
            }
        }
        out += to_string(ans);
        out += '\n';
    }
    cout << out;
    return 0;
}
```

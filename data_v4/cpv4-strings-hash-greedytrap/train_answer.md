**Problem.** A *square* is a string of the form `uu` (even length `>= 2`, first half equal to second
half: `aa`, `abab`, ...). Given a string `s` of length `n` (`0 <= n <= 5000`), select
non-overlapping substrings of `s`, each of which is a square, to **maximize the total number of
characters covered**. The empty selection covers `0`, so the answer is at least `0`. Read `n` and
(if `n > 0`) the string from stdin; print the maximum coverage.

**Why the obvious greedy is wrong.** "Scan left to right; at the first position where a square
begins, grab one (shortest, or longest) and jump past it" fails because squares interlock: a short
square starting at position `p` overlaps a longer square starting at `p+1`, and taking the short one
forbids the long one. On `s = "aabab"` the only squares are `aa` at `[0,2)` and `abab` at `[1,5)`,
overlapping at position `1`. Greedy grabs `aa` (coverage `2`) and is then stuck on `bab`; the optimum
skips `aa` and takes `abab` for coverage `4`. Greedy is discarded.

**Key idea — prefix DP with a hashed `O(1)` square test.** Let `dp[i]` = maximum coverage using
non-overlapping squares lying entirely inside `s[0..i)`. The state is just the prefix length, because
non-overlap is enforced automatically: a square ending at `i` consumes `[i-2L, i)` and the rest of
the coverage comes from `dp[i-2L]`, which only uses positions `< i-2L`. Transition:

- `dp[i] = dp[i-1]` (leave position `i-1` uncovered), and
- for every `L >= 1` with `2L <= i` such that `s[i-2L..i-L) == s[i-L..i)`,
  `dp[i] = max(dp[i], dp[i-2L] + 2L)`.

Answer: `dp[n]`. Testing whether a window `s[i-2L..i)` is a square is "do its two length-`L` halves
match", which a polynomial rolling hash answers in `O(1)`. Total `O(n^2)` time, `O(n)` memory — fine
for `n = 5000` (about `1.25*10^7` hash comparisons).

**Pitfalls.**
1. *Greedy.* Local "take a square now" can forbid more coverage than it gains (`aabab`). Use the DP.
2. *Unsigned underflow in the substring hash.* `sub(l,r) = h[r] - h[l]*base^{r-l}` computed in
   unsigned 64-bit can wrap when `h[r] < h[l]*base^{r-l} (mod MOD)`. Add `MOD` first:
   `res = h[r] + MOD - x; if (res >= MOD) res -= MOD;`. Otherwise equal substrings can be reported
   unequal and the DP silently mis-counts.
3. *Off-by-one in the square test.* The second half of a square ending at `i` is `s[i-L..i)`, length
   exactly `L`. Comparing against `s[i-L..i+1)` (length `L+1`) makes every length-`2` square
   undetectable, so the answer collapses (e.g. `aaaa -> 0`). End the second half at `i`.
4. *Hash soundness.* A single 61-bit modulus never reports a false negative; false positives are
   `~1/MOD` per comparison, negligible over `O(n^2)`. Map characters to `>= 1` so leading "zeros"
   cannot collide distinct strings; randomize the base from a fixed seed to dodge anti-hash inputs.

**Edge cases.** `n = 0` (read no string, print `0`); `n = 1` (no even window, `0`); a string with no
square (`abc -> 0`); all-equal strings (`aaaaaa -> 6`, whole string tiled). The `O(1)` answer fits in
`int`, but accumulate hashes in 64-bit.

**Complexity.** `O(n^2)` time, `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    string s;
    if (n > 0) cin >> s;                   // n == 0: no string token follows

    // Defensive: trust n as the authoritative length.
    if ((int)s.size() != n) n = (int)s.size();

    // Polynomial rolling hash with a single 64-bit modulus, base chosen at random
    // from a fixed seed range to dodge anti-hash inputs. h[i] = hash of s[0..i-1].
    // pw[k] = base^k mod MOD. We compare two equal-length substrings in O(1) by
    // their hash; a length-2L window s[j..j+2L) is a "square" iff its two halves
    // s[j..j+L) and s[j+L..j+2L) have equal hash.
    const unsigned long long MOD = (1ULL << 61) - 1; // Mersenne prime 2^61 - 1
    auto mulmod = [&](unsigned long long a, unsigned long long b) -> unsigned long long {
        // 128-bit multiply then reduce modulo 2^61 - 1.
        __uint128_t c = (__uint128_t)a * b;
        unsigned long long lo = (unsigned long long)(c & MOD);
        unsigned long long hi = (unsigned long long)(c >> 61);
        unsigned long long r = lo + hi;
        if (r >= MOD) r -= MOD;
        return r;
    };

    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    unsigned long long base = (rng() % (MOD - 256)) + 256; // in [256, MOD)

    vector<unsigned long long> h(n + 1, 0), pw(n + 1, 1);
    for (int i = 0; i < n; i++) {
        unsigned long long c = (unsigned long long)(unsigned char)s[i] + 1; // map char -> [1..256]
        h[i + 1] = mulmod(h[i], base) + c;
        if (h[i + 1] >= MOD) h[i + 1] -= MOD;
        pw[i + 1] = mulmod(pw[i], base);
    }
    // hash of s[l..r) (0-indexed, half-open), length r-l
    auto sub = [&](int l, int r) -> unsigned long long {
        // h[r] - h[l]*base^(r-l)
        unsigned long long x = mulmod(h[l], pw[r - l]);
        unsigned long long res = h[r] + MOD - x;
        if (res >= MOD) res -= MOD;
        return res;
    };

    // dp[i] = maximum total length coverable by non-overlapping squares inside s[0..i).
    // Transition: dp[i] = dp[i-1], and for every even length 2L with 2L <= i such that
    // s[i-2L..i) is a square, dp[i] = max(dp[i], dp[i-2L] + 2L).
    vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i - 1];
        // square of length 2L ends at i, starts at j = i - 2L, halves at j and j+L
        for (int L = 1; 2 * L <= i; L++) {
            int j = i - 2 * L;
            if (sub(j, j + L) == sub(j + L, i)) {
                int cand = dp[j] + 2 * L;
                if (cand > dp[i]) dp[i] = cand;
            }
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
```

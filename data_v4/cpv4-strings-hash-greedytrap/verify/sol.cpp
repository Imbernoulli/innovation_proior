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

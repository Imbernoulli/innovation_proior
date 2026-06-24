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

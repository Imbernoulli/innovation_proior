// chk.cc -- fsx_A_0996 : Balanced Word Maximal Factor Richness
// Feasibility: exact length/alphabet, whole-word letter counts within tol_g of
// freq[], and every window of length m=1..K has, for every letter c, a count
// within [floor(m*freq_c/L)-tol_w, ceil(m*freq_c/L)+tol_w] (clamped to [0,m]).
// Objective (only if feasible): F = FactorScore + w_pal * PalScore where
// FactorScore = sum_{l=1..K} #distinct length-l factors, PalScore = #distinct
// palindromic factors of length 1..K. Internal baseline B = F() of the
// checker's own short-period (period<=6) construction matching freq[] exactly.
#include "testlib.h"
#include <vector>
#include <string>
#include <unordered_set>
#include <cstdint>

using namespace std;
typedef unsigned long long u64;
typedef long long ll;

// ---- double hashing for O(1) substring compare ----
static const u64 MOD1 = 1000000000000000003ULL; // just used as a big odd modulus via mulmod trick
static const u64 BASE1 = 131542391;
static const u64 BASE2 = 1000000009;
static const unsigned MOD2 = 998244353;
static const unsigned MOD3 = 1000000007;

struct Hasher {
    int n;
    vector<unsigned> h1, h2, p1, p2; // mod MOD2, MOD3 with base BASE1/BASE2 (both fit in 32-bit range via mod)
    void build(const string& s) {
        n = (int)s.size();
        h1.assign(n + 1, 0); h2.assign(n + 1, 0);
        p1.assign(n + 1, 1); p2.assign(n + 1, 1);
        for (int i = 0; i < n; i++) {
            h1[i + 1] = (unsigned)((1ULL * h1[i] * BASE1 + (unsigned char)s[i] + 1) % MOD2);
            h2[i + 1] = (unsigned)((1ULL * h2[i] * BASE2 + (unsigned char)s[i] + 1) % MOD3);
            p1[i + 1] = (unsigned)((1ULL * p1[i] * BASE1) % MOD2);
            p2[i + 1] = (unsigned)((1ULL * p2[i] * BASE2) % MOD3);
        }
    }
    // hash of s[l, l+len)
    inline u64 get(int l, int len) const {
        unsigned a = (unsigned)((h1[l + len] + 1ULL * MOD2 - (1ULL * h1[l] * p1[len]) % MOD2) % MOD2);
        unsigned b = (unsigned)((h2[l + len] + 1ULL * MOD3 - (1ULL * h2[l] * p2[len]) % MOD3) % MOD3);
        return (u64)a * 2000000011ULL + b;
    }
};

// F(w, K, w_pal): FactorScore + w_pal * PalScore, factor/palindrome lengths 1..K
static ll scoreWord(const string& w, int K, int wpal) {
    int L = (int)w.size();
    Hasher fwd; fwd.build(w);
    string rev(w.rbegin(), w.rend());
    Hasher bwd; bwd.build(rev);

    unordered_set<u64> factors;
    unordered_set<u64> pals;
    factors.reserve((size_t)K * 4 + 16);
    pals.reserve((size_t)K * 4 + 16);

    for (int l = 1; l <= K && l <= L; l++) {
        for (int i = 0; i + l <= L; i++) {
            u64 hf = fwd.get(i, l);
            // combine length into key to avoid cross-length collisions
            u64 key = hf * 1000003ULL + (u64)l;
            factors.insert(key);
            // palindrome check: s[i,i+l) reversed equals s at mirrored position in rev
            int j = L - i - l; // start index in rev of the reversed substring
            u64 hb = bwd.get(j, l);
            if (hf == hb) {
                pals.insert(key);
            }
        }
    }
    return (ll)factors.size() + (ll)wpal * (ll)pals.size();
}

// exact rational deficit construction with denominator `den`, target ratios num[]/den
static string deficitRational(ll L, int a, const vector<ll>& num, ll den) {
    vector<ll> cnt(a, 0);
    string w(L, '0');
    for (ll i = 0; i < L; i++) {
        int best = 0;
        ll bestScore = num[0] * (i + 1) - cnt[0] * den;
        for (int c = 1; c < a; c++) {
            ll sc = num[c] * (i + 1) - cnt[c] * den;
            if (sc > bestScore) { bestScore = sc; best = c; }
        }
        cnt[best]++;
        w[i] = (char)('0' + best);
    }
    return w;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int a = inf.readInt(2, 4, "a");
    ll L = inf.readLong(1LL, 5000000LL, "L");
    int K = inf.readInt(1, 200, "K");
    int tol_w = inf.readInt(0, 50, "tol_w");
    int tol_g = inf.readInt(0, 200, "tol_g");
    int w_pal = inf.readInt(0, 100, "w_pal");
    vector<ll> freq(a);
    ll sumFreq = 0;
    for (int c = 0; c < a; c++) { freq[c] = inf.readLong(0, L, "freq_c"); sumFreq += freq[c]; }

    // ---- read participant output: a single token, the word ----
    string w = ouf.readToken();
    if ((ll)w.size() != L) quitf(_wa, "output length %d != L=%lld", (int)w.size(), L);
    vector<ll> cnt(a, 0);
    for (char ch : w) {
        if (ch < '0' || ch > '0' + a - 1) quitf(_wa, "character '%c' not in alphabet [0,%d]", ch, a - 1);
        cnt[ch - '0']++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after the word");

    // ---- global frequency constraint (within tol_g) ----
    for (int c = 0; c < a; c++) {
        ll d = cnt[c] - freq[c];
        if (d < 0) d = -d;
        if (d > tol_g) quitf(_wa, "letter %d count %lld deviates from target %lld by more than tol_g=%d", c, cnt[c], freq[c], tol_g);
    }

    // ---- windowed balance constraint (within tol_w) for m = 1..K ----
    for (int m = 1; m <= K; m++) {
        vector<ll> winCnt(a, 0), winMin(a, -1), winMax(a, -1);
        for (int c = 0; c < a; c++) { winMin[c] = m + 1; winMax[c] = -1; }
        for (int i = 0; i < (int)w.size(); i++) {
            winCnt[w[i] - '0']++;
            if (i >= m) winCnt[w[i - m] - '0']--;
            if (i >= m - 1) {
                for (int c = 0; c < a; c++) {
                    if (winCnt[c] < winMin[c]) winMin[c] = winCnt[c];
                    if (winCnt[c] > winMax[c]) winMax[c] = winCnt[c];
                }
            }
        }
        for (int c = 0; c < a; c++) {
            // classical balance property: across ALL windows of the SAME length m,
            // the count of letter c may vary by at most tol_w (NOT compared against
            // a fixed "ideal" with independent slack on each side -- that looser form
            // admits high-entropy near-random words that are not actually balanced).
            if (winMax[c] - winMin[c] > tol_w) {
                quitf(_wa, "window length %d letter %d out of balance: min=%lld max=%lld spread=%lld > tol_w=%d",
                      m, c, winMin[c], winMax[c], winMax[c] - winMin[c], tol_w);
            }
        }
    }

    // ---- feasible: compute objective ----
    ll F = scoreWord(w, K, w_pal);

    // ---- internal baseline B: short-period (<=12) construction matching freq[] ----
    const ll P0 = 6;
    vector<ll> num(a, 0);
    {
        vector<pair<double,int>> frac(a);
        ll s = 0;
        for (int c = 0; c < a; c++) {
            double target = (double)freq[c] * (double)P0 / (double)L;
            num[c] = (ll)floor(target);
            frac[c] = {target - floor(target), c};
            s += num[c];
        }
        sort(frac.rbegin(), frac.rend());
        ll rem = P0 - s;
        for (int k = 0; k < (int)frac.size() && rem > 0; k++) { num[frac[k].second]++; rem--; }
    }
    string bw = deficitRational(L, a, num, P0);
    ll B = scoreWord(bw, K, w_pal);
    if (B < 1) B = 1;

    double sc = 100.0 * (double)F / (double)B;
    if (sc > 1000.0) sc = 1000.0;
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}

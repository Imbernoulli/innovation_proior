#include <bits/stdc++.h>
using namespace std;

// Count integers x in [1, N] (N >= 0) whose decimal digit sum is divisible by
// the number of decimal digits of x (its length, no leading zeros).
//
// The divisor depends on the length of x, which is non-local, so we cannot run
// one digit DP with a single modulus. Instead we group by length: for every
// target length len (1..19) the modulus is exactly len and fixed, so a standard
// digit DP over numbers of that length works. We count, for each len, how many
// numbers with EXACTLY len digits (leading digit 1..9) lie in [1, N] and have
// digit sum divisible by len.

typedef long long ll;

// Count numbers of exactly `len` digits, value <= bound, no leading zero,
// digit sum divisible by `mod` (== len). `bd` is the digit array of the upper
// bound (most significant first) of length exactly `len`. Returns the count.
//
// Technique: tight-prefix walk + precomputed free-suffix counts. We precompute
// suf[p][s] = number of ways to fill positions p..len-1 with free digits 0..9
// whose own digit sum is ≡ s (mod). Then we walk the tight prefix; at each
// position, every digit strictly below the bound digit "releases" a free suffix,
// whose count we read straight out of suf[]. Finally we add the bound itself if
// it qualifies.
ll countLen(int len, int mod, const vector<int>& bd) {
    int npos = len;
    // suf[p][s] = number of digit strings for positions p..len-1 (each digit
    // 0..9) whose OWN digit sum ≡ s (mod `mod`). Recurrence: placing digit d at
    // position p needs the remaining suffix (p+1..) to have digit sum ≡ s-d.
    vector<vector<ll>> suf(npos + 1, vector<ll>(mod, 0));
    suf[npos][0] = 1; // empty suffix has digit sum 0
    for (int p = npos - 1; p >= 0; --p) {
        for (int s = 0; s < mod; ++s) {
            ll acc = 0;
            for (int d = 0; d <= 9; ++d) {
                int ns = ((s - d) % mod + mod) % mod;
                acc += suf[p + 1][ns];
            }
            suf[p][s] = acc;
        }
    }
    // Walk the tight path. At position p, residue r (digit sum so far mod `mod`).
    // We must respect: at position 0 the digit range is 1..9 (no leading zero),
    // at positions >0 it is 0..9.
    ll total = 0;
    int r = 0;        // current residue along the tight prefix
    bool feasible = true; // whether the exact-bound path is still alive
    for (int p = 0; p < npos; ++p) {
        int lo = (p == 0) ? 1 : 0;
        int hi = bd[p];
        // place digit d in [lo, hi-1] strictly less than bound digit -> free suffix
        for (int d = lo; d < hi; ++d) {
            int nr = (r + d) % mod;
            int need = (mod - nr) % mod; // suffix residue needed to total 0
            total += suf[p + 1][need];
        }
        // continue tight with d == bd[p], but only if it respects lo
        if (bd[p] < lo) { feasible = false; break; }
        r = (r + bd[p]) % mod;
    }
    if (feasible && r == 0) total += 1; // the bound itself, if its digit sum ≡ 0
    return total;
}

// Count x in [1, N] with digitSum(x) % len(x) == 0.  N >= 0.
ll countUpTo(ll N) {
    if (N <= 0) return 0;
    // decompose N into digits
    string s = to_string(N);
    int D = (int)s.size(); // number of digits of N
    ll ans = 0;
    // For lengths 1 .. D-1: all numbers of that length are fully below N's range
    // upper bound is the maximal len-digit number (all 9s), i.e. unrestricted.
    // For length D: bounded by N.
    for (int len = 1; len <= D; ++len) {
        int mod = len; // length-dependent modulus
        vector<int> bd(len);
        if (len < D) {
            // upper bound = 99...9 (len nines): every len-digit number qualifies
            for (int i = 0; i < len; ++i) bd[i] = 9;
        } else {
            for (int i = 0; i < len; ++i) bd[i] = s[i] - '0';
        }
        ans += countLen(len, mod, bd);
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll L, R;
    if (!(cin >> L >> R)) return 0;
    ll ans = countUpTo(R) - countUpTo(L - 1);
    cout << ans << "\n";
    return 0;
}

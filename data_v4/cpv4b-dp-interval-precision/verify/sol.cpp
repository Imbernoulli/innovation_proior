#include <bits/stdc++.h>
using namespace std;

// Partition planks 0..n-1 into contiguous groups, each group covering 1..K planks.
// Group [l..r]: B = sum of beauties, W = sum of widths (W>0). The group is ADMISSIBLE
// iff its average steepness |B|/W is at least p/q, tested EXACTLY by cross-multiplication
// |B|*q >= p*W (never by floating-point division). Merit of an admissible group is B*B.
// Cover every plank using only admissible groups, maximizing the total merit; if no valid
// partition exists, print IMPOSSIBLE. Sums of squares can exceed 64 bits, so the DP value
// is carried in __int128.

static const __int128 NEG_INF = -((__int128)1 << 120); // unreachable sentinel

// print a (possibly negative) __int128
static string i128_to_string(__int128 v) {
    if (v == 0) return "0";
    bool neg = v < 0;
    // careful: -NEG would overflow only at the true minimum, which we never print
    unsigned __int128 u = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
    string s;
    while (u > 0) { s += char('0' + (int)(u % 10)); u /= 10; }
    if (neg) s += '-';
    reverse(s.begin(), s.end());
    return s;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, K;
    if (!(cin >> n >> K)) return 0;
    vector<long long> b(n), w(n);
    for (long long i = 0; i < n; i++) cin >> b[i] >> w[i];
    long long p, q;
    cin >> p >> q;

    // prefix sums (64-bit is enough: |b|,w <= 1e6, n <= 2e5 -> |pb|,pw <= 2e11)
    vector<long long> pb(n + 1, 0), pw(n + 1, 0);
    for (long long i = 0; i < n; i++) {
        pb[i + 1] = pb[i] + b[i];
        pw[i + 1] = pw[i] + w[i];
    }

    // dp[i] = best total merit covering the first i planks; NEG_INF = unreachable.
    vector<__int128> dp(n + 1, NEG_INF);
    dp[0] = 0;
    for (long long i = 1; i <= n; i++) {
        __int128 best = NEG_INF;
        long long lo = max(0LL, i - K);
        for (long long j = lo; j < i; j++) {
            if (dp[j] == NEG_INF) continue;
            long long B = pb[i] - pb[j];   // |B| <= K*1e6 = 3e8
            long long W = pw[i] - pw[j];   // 0 < W <= K*1e6 = 3e8
            long long absB = B < 0 ? -B : B;
            // EXACT admissibility: |B|*q >= p*W. Both sides reach ~3e17 (fit signed 64-bit),
            // but a floating |B|/W vs p/q test misjudges near-equal ratios; compare products.
            __int128 lhs = (__int128)absB * (__int128)q;
            __int128 rhs = (__int128)p * (__int128)W;
            if (lhs >= rhs) {
                __int128 merit = (__int128)B * (__int128)B; // up to 9e16, fits int64 alone
                __int128 cand = dp[j] + merit;              // SUM reaches ~6e19 -> needs __int128
                if (best == NEG_INF || cand > best) best = cand;
            }
        }
        dp[i] = best;
    }

    if (dp[n] == NEG_INF) cout << "IMPOSSIBLE" << "\n";
    else cout << i128_to_string(dp[n]) << "\n";
    return 0;
}

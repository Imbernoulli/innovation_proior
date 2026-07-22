#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker for "Collapsing Pigeonhole Cabinets" (variable-stride-trie-budget).
//
// Input:  K M / per cabinet c: (D_c W_c) then T_c[0..D_c]   (see statement.txt).
// Output: K lines, cabinet c: m_c  s_1 .. s_{m_c}  (strides, sum == D_c).
//
// mem_c   = sum_j T_c[p_{j-1}] * 2^{s_j}     (p_0=0, p_j=p_{j-1}+s_j)
// total memory sum_c mem_c must be <= M.
// F (objective, MIN) = sum_c W_c * m_c.
// Baseline B: F of the "all strides = 1" (m_c = D_c) do-nothing plan.
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int K = inf.readInt(1, 500, "K");
    ll M = inf.readLong(1, 2000000000LL, "M");

    vector<int> D(K);
    vector<ll> W(K);
    vector<vector<ll>> T(K);

    for (int c = 0; c < K; c++) {
        D[c] = inf.readInt(1, 20, "D_c");
        W[c] = inf.readLong(1LL, 200000LL, "W_c");
        T[c].resize(D[c] + 1);
        T[c][0] = inf.readLong(1LL, 1LL, "T_c[0]");   // must be exactly 1
        for (int d = 1; d <= D[c]; d++) {
            ll hi = 2 * T[c][d - 1];
            if (hi > (1LL << 20)) hi = (1LL << 20);
            T[c][d] = inf.readLong(1, hi, "T_c[d]");
        }
    }

    // ---- baseline B: all strides = 1 for every cabinet ----
    ll B = 0;
    for (int c = 0; c < K; c++) B += W[c] * (ll)D[c];
    if (B <= 0) B = 1;

    // ---- read participant output, validate feasibility, accumulate F and total memory ----
    ll F = 0;
    ll totalMem = 0;
    for (int c = 0; c < K; c++) {
        int m = ouf.readInt(1, D[c], "m_c");
        ll sum = 0;
        ll depth = 0;
        ll memc = 0;
        for (int j = 0; j < m; j++) {
            int s = ouf.readInt(1, D[c], "stride");
            sum += s;
            if (sum > D[c]) quitf(_wa, "cabinet strides overshoot D_c");
            memc += T[c][depth] * (1LL << s);
            if (memc < 0 || memc > (ll)4e15) quitf(_wa, "memory overflow / absurd value");
            depth += s;
        }
        if (sum != D[c]) quitf(_wa, "cabinet strides do not sum to D_c");
        totalMem += memc;
        if (totalMem < 0 || totalMem > (ll)4e15) quitf(_wa, "total memory overflow / absurd value");
        F += W[c] * (ll)m;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (totalMem > M) quitf(_wa, "budget exceeded: mem=%lld > M=%lld", totalMem, M);
    if (F <= 0) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld mem=%lld M=%lld Ratio: %.6f", F, B, totalMem, M, sc / 1000.0);
    return 0;
}

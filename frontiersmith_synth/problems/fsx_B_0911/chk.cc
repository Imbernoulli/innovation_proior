#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Ducting Days: Channels That Survive the Sweep"
// (ducting-channel-gradient).
//
// Input: N C K ; positions x_1..x_N ; M0 base edges (i j w) ;
//        then for s=1..K: M_s duct-scenario-s edges (i j w).
//
// Output (participant): N integers c_1..c_N, each in [1,C].
//
// Objective (MIN): F = max_{s=1..K} ( base_conflict + duct_conflict(s) )
//   where conflict = sum of edge weights whose endpoints share a channel.
//
// Baseline B (checker-computed do-nothing): round-robin c_i = 1+((i-1) mod C),
//   objective F evaluated the same way on the SAME edges.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int C = inf.readInt();
    int K = inf.readInt();
    for (int i = 0; i < N; i++) inf.readLong(); // positions (unused for scoring)

    int M0 = inf.readInt();
    vector<int> bi(M0), bj(M0);
    vector<ll> bw(M0);
    for (int e = 0; e < M0; e++) {
        bi[e] = inf.readInt(1, N, "base_i");
        bj[e] = inf.readInt(1, N, "base_j");
        bw[e] = inf.readLong(1, (ll)2e9, "base_w");
    }

    vector<vector<int>> di(K + 1), dj(K + 1);
    vector<vector<ll>> dw(K + 1);
    for (int s = 1; s <= K; s++) {
        int Ms = inf.readInt();
        di[s].resize(Ms); dj[s].resize(Ms); dw[s].resize(Ms);
        for (int e = 0; e < Ms; e++) {
            di[s][e] = inf.readInt(1, N, "duct_i");
            dj[s][e] = inf.readInt(1, N, "duct_j");
            dw[s][e] = inf.readLong(1, (ll)2e9, "duct_w");
        }
    }

    // ---- internal baseline B: round-robin assignment on the SAME edges ----
    vector<int> rr(N + 1);
    for (int i = 1; i <= N; i++) rr[i] = 1 + ((i - 1) % C);
    ll baseConflictRR = 0;
    for (int e = 0; e < M0; e++)
        if (rr[bi[e]] == rr[bj[e]]) baseConflictRR += bw[e];
    ll B = 0;
    for (int s = 1; s <= K; s++) {
        ll dc = 0;
        for (size_t e = 0; e < di[s].size(); e++)
            if (rr[di[s][e]] == rr[dj[s][e]]) dc += dw[s][e];
        B = max(B, baseConflictRR + dc);
    }
    if (B < 1) B = 1;

    // ---- read participant assignment (strict feasibility) ----
    vector<int> c(N + 1);
    for (int i = 1; i <= N; i++) c[i] = ouf.readInt(1, C, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output after channel assignment");

    // ---- objective F ----
    ll baseConflict = 0;
    for (int e = 0; e < M0; e++)
        if (c[bi[e]] == c[bj[e]]) baseConflict += bw[e];
    ll F = 0;
    for (int s = 1; s <= K; s++) {
        ll dc = 0;
        for (size_t e = 0; e < di[s].size(); e++)
            if (c[di[s][e]] == c[dj[s][e]]) dc += dw[s][e];
        F = max(F, baseConflict + dc);
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}

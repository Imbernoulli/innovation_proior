#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Pick a team that clicks but does not crowd"
// (family: synergy-congestion-team).
//
// Input:  N M
//         v_0 .. v_{N-1}
//         cap_0 .. cap_{N-1}
//         pen_0 .. pen_{N-1}
//         M lines: i j s   (undirected chemistry edge, i<j, s>=1)
//
// Output (participant): k, then k distinct indices in [0,N-1] -- the team S.
//
// Objective (MAX): for i in S let deg_S(i) = #neighbours of i (per the edge
// list) that are also in S.
//   raw = sum_{i in S} v_i + sum_{(i,j) in E, i,j in S} s_ij
//         - sum_{i in S} pen_i * max(0, deg_S(i) - cap_i)
//   F = max(0, raw)
//
// Baseline B (checker-computed, do-nothing-clever): B = sum of v_i over all
// nodes with ZERO edges in the whole graph -- always safe to take (no
// synergy, no crowding possible), so it is a genuinely feasible reference
// construction. B is guaranteed positive by the generator.
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<ll> v(N), cap(N), pen(N);
    for (int i = 0; i < N; i++) v[i] = inf.readLong();
    for (int i = 0; i < N; i++) cap[i] = inf.readLong();
    for (int i = 0; i < N; i++) pen[i] = inf.readLong();
    vector<array<int,3>> edges(M);
    vector<int> totalDeg(N, 0);
    for (int e = 0; e < M; e++) {
        int a = inf.readInt(0, N - 1, "i");
        int b = inf.readInt(0, N - 1, "j");
        int s = inf.readInt(1, 1000000, "s");
        edges[e] = {a, b, s};
        totalDeg[a]++; totalDeg[b]++;
    }

    // ---- internal baseline B: value of all zero-degree (isolated) nodes ----
    ll B = 0;
    for (int i = 0; i < N; i++) if (totalDeg[i] == 0) B += v[i];
    if (B < 1) B = 1;

    // ---- read participant team (strict feasibility) ----
    int k = ouf.readInt(0, N, "k");
    vector<char> chosen(N, 0);
    for (int t = 0; t < k; t++) {
        int idx = ouf.readInt(0, N - 1, "idx");
        if (chosen[idx]) quitf(_wa, "duplicate team member %d", idx);
        chosen[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after team list");

    // ---- evaluate the true objective on the graph ----
    ll valueSum = 0;
    for (int i = 0; i < N; i++) if (chosen[i]) valueSum += v[i];

    ll synergySum = 0;
    vector<ll> degS(N, 0);
    for (int e = 0; e < M; e++) {
        int a = edges[e][0], b = edges[e][1], s = edges[e][2];
        if (chosen[a] && chosen[b]) {
            synergySum += s;
            degS[a]++; degS[b]++;
        }
    }

    ll penaltySum = 0;
    for (int i = 0; i < N; i++) if (chosen[i]) {
        ll excess = degS[i] - cap[i];
        if (excess > 0) penaltySum += pen[i] * excess;
    }

    ll raw = valueSum + synergySum - penaltySum;
    ll F = max(0LL, raw);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld value=%lld synergy=%lld penalty=%lld B=%lld Ratio: %.6f",
          F, valueSum, synergySum, penaltySum, B, sc / 1000.0);
    return 0;
}

#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer -- "Eco-Industrial Park: Waste-Exchange Loops".
//
// Input:  N M ; then M lines  u v cap rev.
// Output: K ; then K lines  e f  (route flow f on pipe e; distinct e; unlisted = 0).
//
// Feasibility (STRICT): 1<=f<=cap_e, distinct pipe indices, and MATERIAL BALANCE
//   (inflow==outflow) at every plant. Any violation -> score 0.
//
// Objective (MAX): F = sum_e rev_e * x_e.
//
// Baseline B (checker-computed, bilateral): for every unordered pair {u,v} that has
//   pipes both ways, take the highest-rev pipe each direction; if their net rev is
//   positive, add net * min(cap_fwd, cap_bwd). B = sum over all such profitable
//   2-cycles. This is exactly the "best-pair matching" plan and what trivial.cpp
//   reproduces (-> ratio ~0.1).
//
// Score (max): sc = 100 * F / B, clamped to [0,1000]; ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<int> eu(M + 1), ev(M + 1);
    vector<ll> ecap(M + 1), erev(M + 1);
    for (int e = 1; e <= M; e++) {
        eu[e]   = inf.readInt();
        ev[e]   = inf.readInt();
        ecap[e] = inf.readLong();
        erev[e] = inf.readLong();
    }

    // ---- internal baseline B: best bilateral (2-cycle) plan ----
    map<pair<int,int>, pair<ll,ll>> best;  // (u,v) -> (maxRev, capOfThatPipe)
    for (int e = 1; e <= M; e++) {
        auto key = make_pair(eu[e], ev[e]);
        auto it = best.find(key);
        if (it == best.end()) best[key] = {erev[e], ecap[e]};
        else if (erev[e] > it->second.first ||
                 (erev[e] == it->second.first && ecap[e] > it->second.second))
            it->second = {erev[e], ecap[e]};
    }
    ll B = 0;
    for (auto& kv : best) {
        int u = kv.first.first, v = kv.first.second;
        if (u < v) {
            auto it = best.find(make_pair(v, u));
            if (it != best.end()) {
                ll net = kv.second.first + it->second.first;
                if (net > 0) B += net * min(kv.second.second, it->second.second);
            }
        }
    }
    if (B <= 0) B = 1;  // generator guarantees profitable 2-cycles -> B>0

    // ---- read & validate participant circulation ----
    int K = ouf.readInt(0, M, "K");
    vector<char> seen(M + 1, 0);
    vector<ll> net(N + 1, 0);
    ll F = 0;
    for (int i = 0; i < K; i++) {
        int e = ouf.readInt(1, M, "edge");
        if (seen[e]) quitf(_wa, "pipe %d listed more than once", e);
        seen[e] = 1;
        ll fl = ouf.readLong(1, ecap[e], "flow");
        F += erev[e] * fl;
        net[eu[e]] -= fl;
        net[ev[e]] += fl;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    for (int i = 1; i <= N; i++)
        if (net[i] != 0)
            quitf(_wa, "material balance violated at plant %d (net=%lld)", i, net[i]);

    double sc = 100.0 * (double)F / (double)B;
    if (sc < 0) sc = 0;
    if (sc > 1000) sc = 1000;
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}

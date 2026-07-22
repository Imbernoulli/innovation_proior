#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Firebreaks Ahead of the Blaze"  (generator)   family: staged-firebreak-forecaster
//
// Composes deterministic-cascade + reachability-frontier + firewall-timing:
//
//   * A SPINE of Dmax+1 rungs, each rung = Wd worthless (weight-0) stands, with
//     consecutive rungs FULLY connected (Wd x Wd). Rung 0 = the sources. Because the
//     rungs are dense and Wd > B, the fire fills a whole rung every step and the front
//     is always WIDER than the per-step budget -> you can NEVER wall off the frontier.
//     A whole rung (size Wd) is the cheapest spine cut, still costlier than a gateway.
//
//   * Modules (groves). Module i hangs off spine rung d_i by a GATEWAY (vertex cut) of
//     C_i weight-0 stands, each wired to all Wd stands of rung d_i; behind the gateway a
//     GROVE of G_i valued stands each wired to ALL C_i gateway stands. The gateway is the
//     UNIQUE minimum cut isolating the grove (size C_i < Wd). Rung d_i burns at step d_i,
//     so the gateway stands burn at step d_i+1 (their BFS distance). To save the grove
//     every gateway stand must be protected before it burns; with budget B that needs
//     ceil(C_i/B) steps of LEAD TIME. HARD modules have C_i > B (must be pre-sealed
//     early); a minority are easy (C_i <= B).
//
//   * A disconnected SAFE set (the only naturally-unreachable value) fixes the baseline
//     B0 > 0, so the trivial do-nothing solution scores exactly 0.1.
//
// TRAP: the obvious "defend the most valuable threatened stands each step" greedy reacts
// only when a grove is already breached -- it rescues at most B grove stands at the last
// instant and seals no gateway with C_i > B, losing the bulk of every distant grove. The
// insight is to read each gateway + its deadline and pre-commit budget by value-density to
// the distant chokes that can still be sealed in time.
//
// Output:  N M B S / weights / sources / M edges.  All randomness seeded from testId.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int B    = 2 + (testId >= 6 ? 1 : 0);          // 2 (tests 1..5) or 3 (tests 6..10)
    int Wd   = 6 + (int)llround(f * 4.0);          // 6..10  front width (> B)
    int nMod = 6 + (int)llround(f * 22.0);         // 6..28  modules
    int Dmax = 5 + (int)llround(f * 25.0);         // 5..30  deepest module rung

    int idc = 0;                                   // vertex id counter (1-based)
    vector<vector<int>> rung(Dmax + 1);
    for (int k = 0; k <= Dmax; k++)
        for (int j = 0; j < Wd; j++) rung[k].push_back(++idc);

    vector<pair<int,int>> edges;
    // spine: fully connect consecutive rungs
    for (int k = 1; k <= Dmax; k++)
        for (int a : rung[k - 1])
            for (int b : rung[k]) edges.push_back({a, b});

    // weights: default 0
    vector<ll> wtmp;                               // grow as ids appended; index by id-1
    wtmp.assign(idc, 0);

    ll totalGroveVal = 0;
    // modules
    for (int i = 0; i < nMod; i++){
        int d = 2 + rnd.next(0, Dmax - 2);         // depth in [2, Dmax]
        bool hard = rnd.next(0, 9) < 7;            // 70% hard (C > B)
        int C = hard ? (B + 1 + rnd.next(0, 3))    // B+1 .. B+4
                     : (1 + rnd.next(0, B - 1));   // 1 .. B
        int Gmax = 10 + (int)llround(f * 55.0);    // 10 .. 65
        int G = 8 + rnd.next(0, Gmax);             // grove size (small -> greedy's last-instant rescue is visible)
        ll unit = 5 + rnd.next(0, 25);             // per-stand value 5..30

        // gateway stands
        vector<int> gate;
        for (int c = 0; c < C; c++){ gate.push_back(++idc); wtmp.push_back(0); }
        for (int c : gate) for (int r : rung[d]) edges.push_back({c, r});
        // grove stands, each wired to every gateway stand
        for (int g = 0; g < G; g++){
            int gv = ++idc; wtmp.push_back(unit); totalGroveVal += unit;
            for (int c : gate) edges.push_back({gv, c});
        }
    }

    // safe set: disconnected, carries the baseline value
    int nSafe = 8;
    ll W_safe = (ll)llround((double)totalGroveVal * 0.13);
    if (W_safe < nSafe) W_safe = nSafe;            // keep each safe stand >= 1
    vector<int> safeIds;
    for (int s = 0; s < nSafe; s++){ safeIds.push_back(++idc); wtmp.push_back(0); }
    // distribute W_safe as evenly as possible
    for (int s = 0; s < nSafe; s++){
        ll share = W_safe / nSafe + (s < (int)(W_safe % nSafe) ? 1 : 0);
        wtmp[safeIds[s] - 1] = share;
    }

    int N = idc;
    int M = (int)edges.size();
    int S = Wd;                                    // sources = rung 0

    // shuffle edge listing so the structure isn't given away by order
    for (int i = M - 1; i > 0; i--) swap(edges[i], edges[rnd.next(0, i)]);

    printf("%d %d %d %d\n", N, M, B, S);
    for (int v = 1; v <= N; v++) printf("%lld%c", wtmp[v - 1], v == N ? '\n' : ' ');
    for (int j = 0; j < Wd; j++) printf("%d%c", rung[0][j], j == Wd - 1 ? '\n' : ' ');
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}

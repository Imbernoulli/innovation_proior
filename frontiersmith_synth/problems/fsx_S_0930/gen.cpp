#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Compressed-Air City Grid: Booster Siting Under Quadratic Line Loss" (generator)
// family: superlinear-flow-splitting
//
// Builds nPairs independent source/sink "hub chains". Each chain is a sequence of
// hub junctions H0(source) -> H1 -> ... -> Hm(sink); consecutive hubs are joined
// by a BUNDLE of p near-parallel pipes (PLANTED structure: equal-ish resistance,
// only one bundle member is booster-ready). A single unsplit path can only ever
// use ONE pipe per bundle, so it pays the full r*amt^2 there, while the true
// current-division equilibrium spreads amt across all p members and pays roughly
// 1/p of that -- the parallel-path-balancing trap.
//
// A handful of high-resistance, high-"looking-leaky" DECOY candidate pipes are
// planted as dead-end pendants off random nodes (no source/sink ever attaches to
// them, so their true equilibrium current is exactly 0 in every feasible
// solution). A naive "boost whichever candidate pipe has the highest raw
// resistance" heuristic (ignoring where current actually flows) wastes its
// booster budget there; only re-solving the network (or otherwise tracking real
// current) reveals they are worthless -- the booster-siting trap.
//
// All hub-chain and decoration nodes are stitched into one connected graph by
// construction (every new node attaches to an existing one).
// -----------------------------------------------------------------------------

struct Edge { int u, v; ll r; int cand; ll gain; };

static vector<Edge> edges;
static vector<pair<int,ll>> srcs, sinks;

static void addEdge(int u, int v, ll r, int cand, ll gain) {
    edges.push_back({u, v, r, cand, gain});
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Fixed tiny instance -- identical to the statement's worked example.
        printf("3 3 1 1 1\n");
        printf("1 12\n");
        printf("3 12\n");
        printf("1 2 8 1 6\n");
        printf("1 2 8 0 0\n");
        printf("2 3 2 0 0\n");
        return 0;
    }

    double f = (testId - 2) / 8.0; // 0 .. 1 over testId=2..10

    int nPairs = 1 + (testId >= 5 ? 1 : 0) + (testId >= 8 ? 1 : 0); // 1,2,3
    int mBase = 2 + (int)llround(f * 4.0);                          // 2..6 stages
    int pmax  = 3 + (int)llround(f * 6.0);                          // 3..9 max bundle width
    bool trapDecoys = (testId == 4 || testId == 6 || testId == 7 || testId == 9 || testId == 10);

    int nextId = 1;
    int totalStages = 0;
    int prevPairHub0 = -1;
    for (int k = 0; k < nPairs; k++) {
        int m = max(1, mBase - k);
        vector<int> hub(m + 1);
        for (int i = 0; i <= m; i++) hub[i] = nextId++;

        ll amt = 30 + (ll)rnd.next(0, (int)llround(60.0 + f * 260.0));

        for (int i = 0; i < m; i++) {
            int p = 2 + rnd.next(0, max(1, pmax - 2));
            int boostSlot = rnd.next(0, p - 1);
            for (int j = 0; j < p; j++) {
                ll r = 4 + rnd.next(0, 21);          // 4..25
                int cand = (j == boostSlot) ? 1 : 0;
                ll gain = cand ? (1 + rnd.next(0, (int)(r - 2))) : 0; // 1 .. r-1
                addEdge(hub[i], hub[i + 1], r, cand, gain);
            }
            totalStages++;
        }
        // Stitch every pair's hub-chain component onto the previous one so the
        // WHOLE graph stays a single connected component (this plain link never
        // carries a pair's own required flow more cheaply than its own chain
        // since it is not on that pair's shortest path unless deliberately
        // useful, but it guarantees reachability for every source/sink lookup).
        if (prevPairHub0 >= 0) {
            ll r = 3 + rnd.next(0, 23);
            addEdge(prevPairHub0, hub[0], r, 0, 0);
        }
        prevPairHub0 = hub[0];
        srcs.push_back({hub[0], amt});
        sinks.push_back({hub[m], amt});
    }

    int hubNodeCount = nextId - 1;
    int extraDec = (int)llround(f * 122.0);
    int V = min(140, hubNodeCount + extraDec);
    extraDec = max(0, V - hubNodeCount);

    int K = max(1, min(8, totalStages));
    int decoysWanted = trapDecoys ? max(K, min(8, K + 2)) : min(2, extraDec / 6 + 1);

    int decoysPlaced = 0;
    for (int id = hubNodeCount + 1; id <= V; id++) {
        int parent = 1 + rnd.next(0, id - 2); // any earlier node
        bool makeDecoy = trapDecoys && decoysPlaced < decoysWanted;
        if (!trapDecoys && decoysPlaced < decoysWanted && rnd.next(0, 3) == 0) makeDecoy = true;
        if (makeDecoy) {
            ll r = 45 + rnd.next(0, 15);           // 45..60, above every real candidate's r (<=25)
            ll gain = 25 + rnd.next(0, 16);        // 25..40
            addEdge(parent, id, r, 1, gain);
            decoysPlaced++;
        } else {
            ll r = 3 + rnd.next(0, 23);            // 3..25, plain decoration pipe
            addEdge(parent, id, r, 0, 0);
        }
    }
    // If size budget was too small to fit every wanted decoy, force at least one more
    // by converting a plain decoration edge on the largest trap tests (rare edge case).
    while (trapDecoys && decoysPlaced < 1 && (int)edges.size() > 0) {
        for (auto &e : edges) {
            if (e.cand == 0 && e.r <= 25) {
                e.r = 45 + rnd.next(0, 15);
                e.gain = 25 + rnd.next(0, 16);
                e.cand = 1;
                decoysPlaced++;
                break;
            }
        }
        break;
    }

    int E = (int)edges.size();
    int S = (int)srcs.size(), T = (int)sinks.size();

    printf("%d %d %d %d %d\n", V, E, S, T, K);
    for (auto &s : srcs) printf("%d %lld\n", s.first, s.second);
    for (auto &s : sinks) printf("%d %lld\n", s.first, s.second);
    for (auto &e : edges) printf("%d %d %lld %d %lld\n", e.u, e.v, e.r, e.cand, e.gain);
    return 0;
}

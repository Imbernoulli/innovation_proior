#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Delay-Line Cascade: Synchronized Toppling Network".
// family: domino-cascade-timing-logic   objective: MAX
//
// Node 1 = source, pulses at tick 0. Candidate track segments (edges, always
// u < v) each carry a propagation delay w and a build cost c. The solver
// COMMITS a subset of segments within budget K (placement-commit). Committed
// segments carry the pulse deterministically (deterministic-cascade): a plain
// node forwards on its EARLIEST arrival; an AND-merge node only fires if at
// least two of its committed inbound segments deliver the pulse on the exact
// same tick (timing-synchronization) -- a lone or mistimed arrival is wasted.
//
// GADGET (the trap): merge node m has two relay predecessors r1, r2.
//   S -> r1 : "cheap" edge (w1c, cc1)     r1 -> m : fixed (rw1, rc1)
//   S -> r2 : "cheap" edge (w2c, cc2)     r2 -> m : fixed (rw2, rc2)
// routing both relays via their CHEAPEST edge (the "shortest distance" reflex)
// gives arrival ticks w1c+rw1 vs w2c+rw2, which the generator deliberately
// makes UNEQUAL for non-easy gadgets. A genuine "delay line" alternate edge on
// the shorter side (weight raised so its tick matches the other branch, at a
// strictly higher cost) is also planted, so a synchronized activation is
// always reachable -- but only by reasoning about exact tick arithmetic, not
// by minimizing per-branch cost/distance. Some gadgets also carry a decoy
// alternate that is off by one tick (looks like a delay line, does not sync).
//
// Independent plain "target" nodes (single direct S-edge, no timing coupling)
// give cheap, reliable partial credit, and noise edges/nodes pad scale without
// contributing value -- a budget-constrained coverage+timing decision.
// -----------------------------------------------------------------------------

struct Edge { int u, v; ll w, c; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    struct Cfg { int gadgets, easyGadgets, targets, noiseNodes, noiseEdges; bool decoys; double kFrac; };
    static const Cfg CFG[11] = {
        {0,0,0,0,0,false,0.0},        // unused index 0
        {1, 1,  3,   4,    4,  false, 0.78},  // 1  tiny warm-up (all-easy sync)
        {2, 1,  3,   6,   10,  false, 0.66},  // 2  first trap appears
        {2, 0,  4,  10,   20,  false, 0.60},  // 3  trap
        {3, 1,  5,  15,   30,  true,  0.60},  // 4  trap + decoys
        {3, 0,  6,  20,   40,  true,  0.55},  // 5  trap
        {1, 0, 14,  40,   80,  true,  0.72},  // 6  NEEDLE: 1 big gadget amid many small targets
        {4, 0,  8,  60,  150,  true,  0.55},  // 7  trap
        {6, 1, 10, 100,  300,  true,  0.52},  // 8  trap, larger
        {8, 1, 14, 200,  700,  true,  0.50},  // 9  trap, larger still
        {14,2, 22, 700, 3500,  true,  0.50},  // 10 largest, fills envelope
    };
    Cfg cfg = CFG[testId];

    vector<int> nodeSync = {0, 0};  // index 0 unused, index 1 = source
    vector<ll>  nodeVal  = {0, 0};
    vector<Edge> edges;

    auto newNode = [&](int sync, ll v) -> int {
        nodeSync.push_back(sync);
        nodeVal.push_back(v);
        return (int)nodeSync.size() - 1;
    };

    ll minTargetCost = -1;
    ll totalDirectCost = 0, totalSyncCost = 0;

    for (int g = 0; g < cfg.gadgets; g++) {
        bool easy = (g < cfg.easyGadgets);
        int r1 = newNode(0, 0), r2 = newNode(0, 0);
        ll mergeVal;
        if (testId == 6) mergeVal = rnd.next(320, 520);   // the needle: big prize
        else mergeVal = rnd.next(60, 150);
        int m = newNode(1, mergeVal);

        ll rw1 = rnd.next(1, 10), rw2 = rnd.next(1, 10);
        ll w1c, w2c;
        if (easy) {
            w1c = rnd.next(10, 30);
            w2c = w1c + rw1 - rw2;              // forces base1 == base2 exactly
            if (w2c < 1) w2c = 1;                // safety clamp (rarely triggers)
        } else {
            w1c = rnd.next(10, 30);
            int tries = 0;
            do { w2c = rnd.next(10, 30); tries++; }
            while (w1c + rw1 == w2c + rw2 && tries < 200);
            if (w1c + rw1 == w2c + rw2) w2c = w1c + 4;  // fallback break-tie
        }
        ll cc1 = rnd.next(3, 10), cc2 = rnd.next(3, 10);
        ll rc1 = rnd.next(1, 4), rc2 = rnd.next(1, 4);

        edges.push_back({1, r1, w1c, cc1});
        edges.push_back({1, r2, w2c, cc2});
        edges.push_back({r1, m, rw1, rc1});
        edges.push_back({r2, m, rw2, rc2});

        ll base1 = w1c + rw1, base2 = w2c + rw2;
        ll trueSync = cc1 + cc2 + rc1 + rc2;

        if (base1 != base2) {
            if (base1 < base2) {
                ll altW = base2 - rw1;
                ll extra = rnd.next(2, 8);
                ll altC = cc1 + extra;
                edges.push_back({1, r1, altW, altC});
                trueSync += extra;
                if (cfg.decoys) {
                    ll dW = altW + (rnd.next(0, 1) ? 1 : -1);
                    if (dW >= 1 && dW != altW && (dW + rw1) != base2) {
                        ll dC = max(1LL, altC - 1);
                        edges.push_back({1, r1, dW, dC});
                    }
                }
            } else {
                ll altW = base1 - rw2;
                ll extra = rnd.next(2, 8);
                ll altC = cc2 + extra;
                edges.push_back({1, r2, altW, altC});
                trueSync += extra;
                if (cfg.decoys) {
                    ll dW = altW + (rnd.next(0, 1) ? 1 : -1);
                    if (dW >= 1 && dW != altW && (dW + rw2) != base1) {
                        ll dC = max(1LL, altC - 1);
                        edges.push_back({1, r2, dW, dC});
                    }
                }
            }
        }
        totalSyncCost += trueSync;
    }

    for (int t = 0; t < cfg.targets; t++) {
        ll v = rnd.next(5, 45);
        int node = newNode(0, v);
        ll w = rnd.next(1, 50);
        ll c = rnd.next(2, 12);
        edges.push_back({1, node, w, c});
        totalDirectCost += c;
        if (minTargetCost < 0 || c < minTargetCost) minTargetCost = c;
    }

    for (int i = 0; i < cfg.noiseNodes; i++) newNode(0, 0);

    int N = (int)nodeSync.size() - 1;
    // Noise edges never target an AND-merge node: a merge's relay set is
    // discovered by the solutions purely from its (non-source) inbound
    // edges, and an unrelated stray inbound edge would corrupt that
    // discovery (silently swap in a bogus "relay"). Restricting the noise
    // edges' head to non-merge nodes keeps every gadget's structure clean.
    for (int i = 0; i < cfg.noiseEdges && N >= 2; i++) {
        int u = -1, v = -1;
        for (int tries = 0; tries < 20; tries++) {
            int uu = rnd.next(1, N - 1);
            int vv = rnd.next(uu + 1, N);
            if (!nodeSync[vv]) { u = uu; v = vv; break; }
        }
        if (u < 0) continue;
        ll w = rnd.next(1, 60);
        ll c = rnd.next(1, 15);
        edges.push_back({u, v, w, c});
    }

    ll K = (ll)llround(cfg.kFrac * (double)(totalDirectCost + totalSyncCost));
    if (minTargetCost > 0) K = max(K, minTargetCost);
    if (K < 1) K = 1;

    // deterministic shuffle of edge order (indices carry no structural meaning)
    for (int i = (int)edges.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(edges[i], edges[j]);
    }

    int M = (int)edges.size();
    printf("%d %d %lld\n", N, M, K);
    for (int v = 2; v <= N; v++) printf("%d %lld\n", nodeSync[v], nodeVal[v]);
    for (auto& e : edges) printf("%d %d %lld %lld\n", e.u, e.v, e.w, e.c);
    return 0;
}
